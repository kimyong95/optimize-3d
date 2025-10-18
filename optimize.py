import math
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple

import accelerate
import sys
import einops
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
import wandb
from absl import app, flags
from accelerate import Accelerator
from accelerate.utils import set_seed
from ml_collections import config_flags
from PIL import Image
from rewards import ObjectiveEvaluator
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from value_model import ValueModel
from utils import collect_calls, to_trimesh, post_process_mesh

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize.py", "Training configuration.")


def update_parameters(mu, sigma, noise, objective_values):
    ''' minimize score '''

    # noise: (T, B, D)
    # objective_values: (T, B)

    assert noise.shape[0] == objective_values.shape[0] == mu.shape[0] == sigma.shape[0]
    assert noise.shape[1] == objective_values.shape[1]

    T_dim = noise.shape[0]
    B_dim = noise.shape[1]
    D_dim = mu.shape[1]

    mu = mu.clone()
    sigma = sigma.clone()

    lr_mu = 1
    lr_sigma = 1 / math.sqrt(D_dim)

    for t in range(T_dim):
        
        objective_values_t = objective_values[t:].mean(0)
        noise_t = noise[t]
        objective_values_t_normalized = (objective_values_t - objective_values_t.mean()) / objective_values_t.std().clamp(min=1e-8)
        
        objective_values_t_softmaxed = nn.Softmax(dim=0)(-objective_values_t_normalized)

        sigma[t] = 1 / (
            
            1/sigma[t] + lr_sigma * (

                (1/sigma[t])[None,:] * (noise_t - mu[t,None,:]) * (noise_t - mu[t,None,:]) * (1/sigma[t])[None,:] * \
                
                objective_values_t_softmaxed[:,None]
            
            # sum over B
            ).sum(0)
        )
        
        mu[t] = mu[t] - lr_mu * (

            (noise_t - mu[t][None,:]) * \
            
            objective_values_t_normalized[:,None]
        
        # mean over B
        ).mean(0)

    return mu, sigma

def get_noise(mu, sigma, batch_size, device, base_noise=None):
    batch_mu = einops.repeat(mu, 'T D -> T B D', B=batch_size)

    batch_sigma = einops.repeat(sigma, 'T D -> T B D', B=batch_size)

    if base_noise is not None:
        base_noise = einops.repeat(base_noise, 'T B ... -> T B (...)')
        assert base_noise.shape == batch_mu.shape
        batch_noise_original = base_noise
    else:
        batch_noise_original = torch.randn(batch_mu.size(), device=device)
    
    batch_noise = batch_mu + batch_sigma**0.5 * batch_noise_original

    batch_noise_original_norm = batch_noise_original.norm(dim=-1)
    batch_noise_norm = batch_noise.norm(dim=-1)
    batch_noise_projected = batch_noise / batch_noise_norm[:,:,None] * batch_noise_original_norm[:,:,None]

    return batch_noise, batch_noise_projected


class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="optimize-3d",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name, "config": config.to_dict()}}
        )

        set_seed(config.seed, device_specific=True)

        self.pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        self.pipeline.to(self.device)
        
        self.objective_evaluator = ObjectiveEvaluator(objective=config.objective)
        self.ref_mesh = trimesh.load(config.ref_mesh_path)
        self.prompt = config.prompt

        self._init_parameters()

        self.value_model = ValueModel(dimension=self.structure_dimension)
        self.value_model.to(self.device)

        self._init_renderer()

        self._log_code()
        
    @property
    def device(self):
        return self.accelerator.device

    def _log_code(self):

        if not self.accelerator.is_main_process:
            return

        cwd = os.path.abspath(os.getcwd())
        imported_py_files = set()
        for module in sys.modules.values():
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py"):
                abs_path = os.path.abspath(path)
                if abs_path.startswith(cwd):
                    imported_py_files.add(abs_path)

        self.accelerator.get_tracker("wandb").run.log_code(".", include_fn=lambda path: path in imported_py_files)

    def _init_renderer(self):
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(1024, 1024)
        self.renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
        self.renderer.scene.scene.add_directional_light("sun", [-0.5, -0.5, -1.0], [1, 1, 1], 90000.0, True)

    def run(self):
        if self.config.eval_freq > 0:
            self.evaluate(step=0)

        for step in range(1, self.config.optimization_steps+1):
            self.train_step(step)

            if self.config.eval_freq > 0 and (step) % self.config.eval_freq == 0:
                self.evaluate(step)

    def train_step(self, step: int) -> None:
        noise, noise_projected = get_noise(self.mu, self.sigma, self.config.batch_size, self.device)

        meshes, slats, pred_data_trajectory = self.generate(intermediate_noise=noise_projected)
        objective_values = self.objective_evaluator(meshes)
        objective_values = torch.from_numpy(objective_values).to(device=self.device, dtype=torch.float32)

        self.accelerator.wait_for_everyone()
        total_batch_size = self.config.batch_size * self.accelerator.num_processes
        gathered_objective_values = self.accelerator.gather(objective_values)
        assert total_batch_size == len(gathered_objective_values)

        _noise = einops.rearrange(noise, "T B D -> B T D")
        gathered_noise = einops.rearrange(self.accelerator.gather(_noise), "B T D -> T B D")

        _pred_data_trajectory = einops.rearrange(pred_data_trajectory, "T B ... -> B T (...)")
        gathered_pred_data_trajectory = einops.rearrange(self.accelerator.gather(_pred_data_trajectory), "B T D -> T B D")

        all_objective_values = None
        if self.accelerator.is_main_process:
            all_objective_values = torch.zeros((self.config.num_inference_steps, total_batch_size), device=self.device,)
            
            self.value_model.add_model_data(
                x = gathered_pred_data_trajectory[-1],
                y = gathered_objective_values,
            )

            for t in range(self.config.num_inference_steps - 1):
                y, _ = self.value_model.predict(gathered_pred_data_trajectory[t])
                all_objective_values[t] = y
            self.mu, self.sigma = update_parameters(self.mu, self.sigma, gathered_noise, all_objective_values)
        self.mu = accelerate.utils.broadcast(self.mu)
        self.sigma = accelerate.utils.broadcast(self.sigma)

        self.log_objective_metrics(objective_values,step=step,stage="train")

    def evaluate(self, step: int) -> None:
        prompts_idx = self.config.batch_size * self.accelerator.process_index + torch.arange(
            self.config.batch_size, device=self.device
        )
        eval_noise = torch.load("eval_noise/struct_tensor.pt", map_location=self.device)[:, prompts_idx, :]
        prior_noise = eval_noise[0]
        intermidiate_noise = eval_noise[1:]

        _, noise_projected = get_noise(
            self.mu,
            self.sigma,
            self.config.batch_size,
            self.device,
            base_noise=intermidiate_noise,
        )

        meshes, slats, _ = self.generate(
            prior_noise=prior_noise,
            intermediate_noise=noise_projected,
        )

        objective_values = self.objective_evaluator(meshes)
        objective_values = torch.from_numpy(objective_values).to(self.device)
        self.log_meshes(meshes, slats, objective_values, step, stage="eval")

        self.log_objective_metrics(objective_values, step=step, stage="eval")
        self.save_parameters(step)

    @torch.inference_mode()
    def generate(
        self,
        prior_noise: Optional[torch.Tensor] = None,
        intermediate_noise: torch.Tensor = None,
    ) -> Tuple[List[trimesh.Trimesh], List[torch.Tensor]]:
        meshes: List[trimesh.Trimesh] = []
        slats: List[torch.Tensor] = []
        pred_data_trajectory: List[torch.Tensor] = []
        intermediate_noise = self._unflatten_structure(intermediate_noise)


        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": 0.7,
            **({"prior_noise": prior_noise} if prior_noise is not None else {}),
            "intermediate_noise": intermediate_noise,
        }

        cond = self.pipeline.get_cond([self.prompt]*self.config.batch_size)
        with collect_calls(self.pipeline.sparse_structure_sampler.sample) as collected_data:            
            coords = self.pipeline.sample_sparse_structure(cond, self.config.batch_size, sparse_structure_sampler_params)
        pred_data_trajectory = torch.stack(collected_data[0]["return"]["pred_x_0"][1:] + [collected_data[0]["return"]["samples"]])

        slats = self.pipeline.sample_slat(cond, coords)

        meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
        meshes = [ to_trimesh(mesh) for mesh in meshes ]
        processed_meshes = [ post_process_mesh(mesh) for mesh in meshes ] # TODO: remove later

        return meshes, slats, pred_data_trajectory

    def save_parameters(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            return

        parameters = {
            "mu": self.mu,
            "sigma": self.sigma,
        }
        wandb_dir = self.accelerator.get_tracker("wandb").run.dir.removesuffix("/files")
        wandb_dir = os.path.relpath(wandb_dir, os.getcwd())
        os.makedirs(f"{wandb_dir}/checkpoints", exist_ok=True)
        ckpt_path = f"{wandb_dir}/checkpoints/{step}.pt"
        self.accelerator.save(parameters, ckpt_path)

    def log_meshes(
        self,
        meshes: List[trimesh.Trimesh],
        slats: List[torch.Tensor],
        objective_values: torch.Tensor,
        step: int,
        stage: str = "train",
    ) -> None:
        gather_meshes = self.accelerator.gather_for_metrics(meshes)
        gather_slats = self.accelerator.gather_for_metrics(slats)
        gather_objective_values = self.accelerator.gather(objective_values)
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        wandb_dir = self.accelerator.get_tracker("wandb").run.dir.removesuffix("/files")
        wandb_dir = os.path.relpath(wandb_dir, os.getcwd())
        mesh_dir = f"{wandb_dir}/meshes/{stage}/{step:03d}"
        slat_dir = f"{wandb_dir}/slats/{stage}/{step:03d}"
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(slat_dir, exist_ok=True)
        for i, (mesh, slat) in enumerate(zip(gather_meshes, gather_slats)):
            mesh.export(f"{mesh_dir}/{i:02d}.glb")
            with open(f"{slat_dir}/{i:02d}.pkl", "wb") as f:
                pickle.dump(slat.cpu(), f)

        views = {
            "front": {"yaw_deg": 0, "pitch_deg": 10},
            "side": {"yaw_deg": 90, "pitch_deg": 10},
            "angle": {"yaw_deg": 45, "pitch_deg": 20},
        }
        wandb_images = defaultdict(list)

        for view_name, view_param in views.items():
            for idx, (mesh, objective_value) in enumerate(zip(gather_meshes, gather_objective_values)):
                wandb_image = wandb.Image(
                    self.render_photo_open3d(mesh, **view_param),
                    caption=f"i={idx},f={objective_value:.4f}",
                    file_type="jpeg",
                )
                wandb_images[f"{stage}_{view_name}_images"].append(wandb_image)
        wandb_tracker = self.accelerator.get_tracker("wandb")
        wandb_tracker.log(wandb_images, step=step)


    def log_objective_metrics(
        self,
        objective_values: torch.Tensor,
        step: int,
        stage: str,
    ) -> None:
        gathered_objective_values = self.accelerator.gather(objective_values)
        self.accelerator.wait_for_everyone()

        prefix = {
            "train": "",
            "eval": "eval_",
        }
        total_batch_size = self.config.batch_size * self.accelerator.num_processes
        metrics = {
            prefix[stage] + "objective_values_mean": gathered_objective_values.mean().item(),
            prefix[stage] + "objective_values_std": gathered_objective_values.std().item(),
            "objective_evaluations": total_batch_size * step,
            "mu_norm": self.mu.norm().item(),
        }

        self.accelerator.log(metrics, step=step)

    def _init_parameters(self) -> None:
        structure_model = self.pipeline.models["sparse_structure_flow_model"]
        self.structure_resolution = structure_model.resolution
        self.structure_channels = structure_model.in_channels
        self.structure_dimension = self.structure_channels * self.structure_resolution ** 3
        steps = self.config.num_inference_steps
        self.mu = torch.zeros((steps, self.structure_dimension), device=self.device)
        self.sigma = torch.ones((steps, self.structure_dimension), device=self.device)

    def _flatten_structure(self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            tensor,
            "... c r1 r2 r3 -> ... (c r1 r2 r3)",
            c=self.structure_channels,
            r1=self.structure_resolution,
            r2=self.structure_resolution,
            r3=self.structure_resolution,
        )

    def _unflatten_structure(self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            tensor,
            "... (c r1 r2 r3) -> ... c r1 r2 r3",
            c=self.structure_channels,
            r1=self.structure_resolution,
            r2=self.structure_resolution,
            r3=self.structure_resolution,
        )

    def render_photo_open3d(self,
                            mesh,
                            yaw_deg=30.0,
                            pitch_deg=20.0,
                            r=5.0,
                            fov_deg=60.0,
                            base_color=(1.0, 1.0, 1.0, 1.0),
                            bg_color=(1.0, 1.0, 1.0, 1.0)) -> Image.Image:
        """
        Take a 'photo' of a trimesh.Trimesh using Open3D OffscreenRenderer.
        Camera orbits the AABB center via (yaw, pitch, r). Headless-safe.
        Returns: PIL.Image (RGBA or RGB depending on support).
        """

        # ---- Orbit camera around AABB center ----
        center = np.asarray(mesh.bounding_box.centroid, dtype=float)
        yaw   = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        eye = center + np.array([
            r * np.cos(pitch) * np.cos(yaw),
            r * np.cos(pitch) * np.sin(yaw),
            r * np.sin(pitch)
        ], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)

        # ---- trimesh -> open3d mesh ----
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=float, copy=True)),
            triangles=o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32, copy=True))
        )
        vnorm = getattr(mesh, "vertex_normals", None)
        if vnorm is not None and len(vnorm) == len(mesh.vertices):
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(vnorm, dtype=float, copy=True))
        else:
            o3d_mesh.compute_vertex_normals()

        # ---- Renderer & scene ----
        scene = self.renderer.scene
        scene.clear_geometry()

        # Material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = base_color
        if hasattr(mat, "base_roughness"): mat.base_roughness = 0.6
        if hasattr(mat, "base_metallic"):  mat.base_metallic = 0.0
        scene.add_geometry("mesh", o3d_mesh, mat)

        # ---- Camera ----
        aspect = 1.0
        bbox_extent = float(np.linalg.norm(np.asarray(mesh.bounding_box.extents, float)))
        near = max(1e-3, 0.01 * max(1.0, r))
        far  = r + 4.0 * max(1.0, bbox_extent) + 10.0 * near
        scene.camera.set_projection(
            fov_deg, aspect, near, far,
            o3d.visualization.rendering.Camera.FovType.Vertical
        )
        scene.camera.look_at(center, eye, up)


        # ---- Render ----
        o3d_img = self.renderer.render_to_image()

        # Robust conversion to numpy for PIL:
        np_img = np.asarray(o3d_img)
        # Ensure contiguous buffer
        np_img = np.ascontiguousarray(np_img)
        # Ensure uint8 (some builds may return float [0,1])
        if np_img.dtype != np.uint8:
            np_img = (np.clip(np_img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        # Create PIL image without 'mode=' kwarg (avoids deprecation warning)
        pil_img = Image.fromarray(np_img)

        return pil_img

if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
