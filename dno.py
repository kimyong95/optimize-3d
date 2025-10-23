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
from torch.utils.checkpoint import checkpoint
from utils import collect_calls, to_trimesh, post_process_mesh

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/treeg.py", "Training configuration.")



class CheckpointWrapper(nn.Module):
    """
    Wraps a module so every forward pass of that module is run under
    torch.utils.checkpoint.checkpoint, saving activation memory.
    """
    def __init__(self, module, use_reentrant=False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant  # PyTorch >=2.0 supports non-reentrant

    def forward(self, *args, **kwargs):
        # checkpoint only accepts tensor args; pack kwargs if any
        if kwargs:
            # pack kwargs into a single tensor tuple to stay compact
            # (keeps this wrapper generic even if blocks add kwargs later)
            def fn(*tensors):
                return self.module(*tensors[:len(args)], **kwargs)
            return checkpoint(fn, *args, use_reentrant=self.use_reentrant)
        else:
            return checkpoint(self.module, *args, use_reentrant=self.use_reentrant)

def enable_gradient_checkpointing(model: nn.Module, use_reentrant: bool = False) -> nn.Module:
    """
    In-place: wraps model.blocks[i] with checkpointing, if not already wrapped.
    Returns the same model for convenience.
    """
    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise ValueError("Model has no .blocks ModuleList to wrap.")

    for i, blk in enumerate(model.blocks):
        if not isinstance(blk, CheckpointWrapper):
            model.blocks[i] = CheckpointWrapper(blk, use_reentrant=use_reentrant)

    # Reflect the setting on the top-level flag if present (harmless otherwise).
    setattr(model, "use_checkpoint", True)
    return model

def disable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    In-place: unwraps checkpoint wrappers and restores original blocks.
    """
    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise ValueError("Model has no .blocks ModuleList to unwrap.")

    for i, blk in enumerate(model.blocks):
        if isinstance(blk, CheckpointWrapper):
            model.blocks[i] = blk.module

    setattr(model, "use_checkpoint", False)
    return model


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
        self.accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
        self.accelerator.init_trackers(
            project_name="optimize-3d",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name, "config": config.to_dict()}}
        )

        set_seed(config.seed, device_specific=True)

        self.pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        [ model[1].requires_grad_(False) for model in self.pipeline.models.items() ] # disable all gradients
        self.pipeline.sample_sparse_structure = self.sample_sparse_structure_dno.__get__(self.pipeline) # replace the original method
        self.pipeline.to(self.device)
        
        self.objective_evaluator = ObjectiveEvaluator(objective=config.objective)
        self.ref_mesh = trimesh.load(config.ref_mesh_path)
        self.prompt = config.prompt

        structure_model = self.pipeline.models["sparse_structure_flow_model"]
        self.structure_resolution = structure_model.resolution
        self.structure_channels = structure_model.in_channels

        self.pipeline.models["sparse_structure_flow_model"] = enable_gradient_checkpointing(self.pipeline.models["sparse_structure_flow_model"])

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

    @torch.no_grad()
    def run(self) -> None:
        all_meshes = []
        all_slats = []
        all_objective_values = []

        for _ in range(self.config.total_num_samples):
            ref_noise = [torch.randn((1, self.structure_channels, self.structure_resolution,self.structure_resolution,self.structure_resolution), device=self.device, requires_grad=True) for _ in range(self.config.num_inference_steps + 1)]
            optimizer = torch.optim.AdamW(ref_noise, lr=0.01, weight_decay=0.0)

            meshes = []
            slats = []
            objective_values = []


            # +1 because we want to log the ref_images after last noise update
            for optimization_i in range(self.config.optimization_steps+1):
                
                # ------------- reference ------------- #
                sparse_structure_sampler_params = {
                    "steps": self.config.num_inference_steps,
                    "noise_level": 0.7,
                    "prior_noise": ref_noise[0],
                    "intermediate_noise": ref_noise[1:],
                }
                cond = self.pipeline.get_cond([self.prompt])
                with torch.enable_grad(), self.accelerator.autocast():
                    ref_coords, ref_xs = self.pipeline.sample_sparse_structure(cond, 1, sparse_structure_sampler_params)
                ref_meshes, ref_slats = self.generate_meshes_from_coords(cond, ref_coords)
                ref_objective_value = self.objective_evaluator(ref_meshes)
                ref_objective_value = torch.tensor(ref_objective_value, device=self.device)
                
                # ------------- logging ref ------------ #
                meshes.append(ref_meshes[0])
                slats.append(ref_slats[0])
                objective_values.append(ref_objective_value)

                # -------------- perturbed ------------- #
                noise = einops.repeat(torch.stack(ref_noise).detach(), "T 1 ... -> T B ...", B=self.config.batch_size)
                noise = noise + torch.randn_like(noise) * 0.1
                sparse_structure_sampler_params = {
                    "steps": self.config.num_inference_steps,
                    "noise_level": 0.7,
                    "prior_noise": noise[0],
                    "intermediate_noise": noise[1:],
                }
                cond = self.pipeline.get_cond([self.prompt]*self.config.batch_size)
                perturbed_coords, perturbed_xs = self.pipeline.sample_sparse_structure(cond, self.config.batch_size, sparse_structure_sampler_params)
                perturbed_meshes, perturbed_slats = self.generate_meshes_from_coords(cond, perturbed_coords)
                perturbed_objective_values = self.objective_evaluator(perturbed_meshes)
                perturbed_objective_values = torch.tensor(perturbed_objective_values, device=self.device)
                
                # ------------ optimization ------------ #
                est_grad = torch.zeros_like(ref_xs[0])
                for i in range(self.config.batch_size):
                    est_grad += (perturbed_objective_values[i] - ref_objective_value) * (perturbed_xs[i] - ref_xs[0])
                est_grad /= (torch.norm(est_grad) + 1e-3)
                with torch.enable_grad():
                    loss = torch.sum(est_grad * ref_xs[0])
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(ref_noise, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            all_meshes.append(meshes)
            all_slats.append(slats)
            all_objective_values.append(objective_values)
        
        # flip: (B, optimization_steps+1) -> (optimization_steps+1, B)
        all_objective_values = list(map(list, zip(*all_objective_values)))
        all_meshes = list(map(list, zip(*all_meshes)))
        all_slats = list(map(list, zip(*all_slats)))

        # log trajectory objective values
        for i in range(self.config.optimization_steps+1):
            objective_values_i = torch.stack(all_objective_values[i])[:,0]
            self.log_objective_metrics(objective_values_i, step=i)
            self.log_meshes(all_meshes[i],all_slats[i],objective_values_i,step=i)

    def generate_meshes_from_coords(self, cond, coords):
        slats = self.pipeline.sample_slat(cond, coords)
        meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
        meshes = [ to_trimesh(mesh) for mesh in meshes ]
        meshes = [ post_process_mesh(mesh) for mesh in meshes ]

        return meshes, slats

    @staticmethod
    def sample_sparse_structure_dno(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        modify the original sample_sparse_structure because we want to collect the xs before masking,
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        if "prior_noise" in sampler_params:
            noise = sampler_params.pop("prior_noise")
        else:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        zs = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        xs = decoder(zs)
        coords = torch.argwhere(xs>0)[:, [0, 2, 3, 4]].int()

        return coords, xs

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
    ) -> None:
        gathered_objective_values = self.accelerator.gather(objective_values)
        self.accelerator.wait_for_everyone()

        metrics = {
            "objective_values_mean": gathered_objective_values.mean().item(),
            "objective_values_std": gathered_objective_values.std().item(),
            "objective_evaluations": self.config.total_num_samples * (self.config.batch_size+1) * step,
        }

        self.accelerator.log(metrics, step=step)

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
