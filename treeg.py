import functools
import math
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple

import accelerate
from easydict import EasyDict as edict
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
from typing import Any
from utils import collect_calls, to_trimesh, post_process_mesh

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/treeg.py", "Training configuration.")


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
        self.pipeline.sparse_structure_sampler.sample_once_original = self.pipeline.sparse_structure_sampler.sample_once
        self.pipeline.sparse_structure_sampler.sample_once = self.sample_once_treeg.__get__(self.pipeline.sparse_structure_sampler)

        self.objective_evaluator = ObjectiveEvaluator(objective=config.objective)
        self.ref_mesh = trimesh.load(config.ref_mesh_path)
        self.prompt = config.prompt

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

    @torch.inference_mode()
    def run(self) -> None:
        all_meshes = []
        all_slats = []

        all_log_trajectory_obj_values = []
        for _ in range(self.config.total_num_samples):
            self.log_trajectory_obj_values = torch.zeros((self.config.num_inference_steps), device=self.device)
            sparse_structure_sampler_params = {
                "steps": self.config.num_inference_steps,
                "noise_level": 0.7,
                "external_self": self,
            }

            cond = self.pipeline.get_cond([self.prompt]*self.config.batch_size)
            coords = self.pipeline.sample_sparse_structure(cond, self.config.batch_size, sparse_structure_sampler_params)
            
            meshes, slats = self.generate_meshes_from_coords(cond, coords)

            all_meshes.extend(meshes)
            all_slats.extend(slats)
            all_log_trajectory_obj_values.append(self.log_trajectory_obj_values)
        
        # log trajectory objective values
        all_log_trajectory_obj_values = torch.stack(all_log_trajectory_obj_values) # (N, T)
        for t in range(self.config.num_inference_steps):
            objective_values_t = all_log_trajectory_obj_values[:, t]
            if torch.isfinite(objective_values_t).all():
                self.log_objective_metrics(objective_values_t, step=t)
        
        self.log_meshes(all_meshes,all_slats,all_log_trajectory_obj_values[:, -1],step=self.config.num_inference_steps-1,stage="eval")

    def generate_meshes_from_coords(self, cond, coords):
        slats = self.pipeline.sample_slat(cond, coords)
        meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
        meshes = [ to_trimesh(mesh) for mesh in meshes ]
        meshes = [ post_process_mesh(mesh) for mesh in meshes ]

        return meshes, slats

    @staticmethod
    def sample_once_treeg(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        noise_level: float = 0.0,
        noise: Optional[torch.Tensor] = None, # this will be ignored
        # ----------- treeg args ----------- #
        external_self = None,
        # ----------- treeg args ----------- #
        **kwargs
    ):
        """
        To replace the original sample_once with treeg sampling
        """
        batch_size = x_t.shape[0]
        rescale_t = external_self.pipeline.sparse_structure_sampler_params["rescale_t"]
        t_seq = np.linspace(1, 0, external_self.config.num_inference_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_idx = np.argwhere(t_seq == t).item()
        t_prev_idx = t_idx + 1
        t_prev_prev_idx = t_idx + 2
        # ----------- original code ----------- #
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        dt = t_prev - t

        std_dev_t = np.sqrt(t / (1 - np.where(t == 1, t_prev, t)))*noise_level
        x_prev_mean = x_t*(1+std_dev_t**2/(2*t)*dt) + pred_v*(1+std_dev_t**2*(1-t)/(2*t))*dt
        
        # ------------ treeg code ------------- #
        x_prev_candidates = []
        x_prev_candidates_obj_values = []
        for i, x_prev_mean_i in enumerate(x_prev_mean):
            # sample
            noise_i = torch.randn( (external_self.config.expansion_size,) + x_prev_mean_i.shape, device=x_prev_mean_i.device)
            x_prev_i = x_prev_mean_i + std_dev_t * np.sqrt(-1*dt) * noise_i
            x_prev_candidates.append(x_prev_i)
            
            # determinisitcally sample once, decode and evaluate
            try:
                pred_sample_i = self.sample_once_original(model, x_prev_i, t_prev, t_seq[t_prev_prev_idx], cond, noise_level=0.0, **kwargs).pred_x_0 if t_prev_prev_idx < len(t_seq) else x_prev_i
                coords = torch.argwhere(external_self.pipeline.models['sparse_structure_decoder'](pred_sample_i)>0)[:, [0, 2, 3, 4]].int()
                cond_dict = external_self.pipeline.get_cond([external_self.prompt]*batch_size)
                meshes, slats = external_self.generate_meshes_from_coords(cond_dict, coords)
                objective_values = external_self.objective_evaluator(meshes)
                objective_values = torch.from_numpy(objective_values).to(x_t.device)
            except Exception as e:
                print(f"Exception {e} when decoding at {t_idx}-th timestep, assign inf objective values")
                objective_values = torch.full((external_self.config.expansion_size,), float('inf'), device=x_t.device)
            
            x_prev_candidates_obj_values.append(objective_values)

        # flatten
        x_prev_candidates = torch.cat(x_prev_candidates, dim=0)
        x_prev_candidates_obj_values = torch.cat(x_prev_candidates_obj_values, dim=0)

        # global selection
        next_indices = x_prev_candidates_obj_values.topk(batch_size, largest=False).indices
        x_prev_candidates_obj_values = x_prev_candidates_obj_values[next_indices]
        external_self.log_trajectory_obj_values[t_idx] = x_prev_candidates_obj_values.mean()
        x_prev = x_prev_candidates[next_indices]
        
        return edict({"pred_x_prev": x_prev, "pred_x_0": pred_x_0})

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
            "front": {"yaw_deg": 180, "pitch_deg": 10},
            "side": {"yaw_deg": 90, "pitch_deg": 10},
            "angle": {"yaw_deg": 135, "pitch_deg": 20},
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
    ) -> None:
        gathered_objective_values = self.accelerator.gather(objective_values)
        self.accelerator.wait_for_everyone()

        total_batch_size = self.config.total_num_samples * self.config.expansion_size
        metrics = {
            "objective_values_mean": gathered_objective_values.mean().item(),
            "objective_values_std": gathered_objective_values.std().item(),
            "objective_evaluations": total_batch_size * (step+1),
        }

        self.accelerator.log(metrics, step=step)

    def render_photo_open3d(self,
                            mesh,
                            yaw_deg=30.0,
                            pitch_deg=20.0,
                            r=4.0,
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
        scene.set_background(bg_color)
        scene.scene.set_sun_light(
            direction=[0.577, -0.577, -0.577], 
            color=[1.0, 1.0, 1.0],             
            intensity=100000                   
        )
        scene.scene.enable_sun_light(True)
        
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
