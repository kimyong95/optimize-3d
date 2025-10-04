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

from rewards import DragCoefficientEstimator
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from value_model import ValueModel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config.py", "Training configuration.")


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

def to_trimesh(mesh_result: MeshExtractResult) -> trimesh.Trimesh:
    vertices = mesh_result.vertices.cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def post_process_mesh(mesh: trimesh.Trimesh, ref_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    - Rotate +90° around X, then +90° around Z (about AABB centroid).
    - Uniformly scale to match reference XY span using s = (sx + sy)/2 (keeps aspect ratio).
    - Translate so XY centers match AND z-min ("floor") equals the reference.
    Returns the same mesh object after in-place transforms.
    """
    if mesh is None or ref_mesh is None:
        return None

    # 1) ROTATE: X +90°, then Z +90°
    c = mesh.bounding_box.centroid
    Rx = trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [1.0, 0.0, 0.0], point=c)
    mesh.apply_transform(Rx)
    c = mesh.bounding_box.centroid
    Rz = trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [0.0, 0.0, 1.0], point=c)
    mesh.apply_transform(Rz)

    # 2) SCALE: compute from bounds AFTER rotations
    ref_bounds = ref_mesh.bounds
    m_bounds = mesh.bounds
    ref_w = float(ref_bounds[1, 0] - ref_bounds[0, 0])
    ref_h = float(ref_bounds[1, 1] - ref_bounds[0, 1])
    m_w   = float(m_bounds[1, 0] - m_bounds[0, 0])
    m_h   = float(m_bounds[1, 1] - m_bounds[0, 1])

    sx = ref_w / m_w
    sy = ref_h / m_h
    s  = (sx + sy) / 2.0  # lock ratio by averaging

    c = mesh.bounding_box.centroid
    S = trimesh.transformations.scale_matrix(float(s), c)
    mesh.apply_transform(S)

    # 3) TRANSLATE: align XY centers and Z floor (z-min) in one shot
    m_bounds = mesh.bounds  # after scaling
    m_cxy = m_bounds.mean(axis=0)[:2]
    r_cxy = ref_bounds.mean(axis=0)[:2]
    dxy = (r_cxy - m_cxy)

    dz = float(ref_bounds[0, 2] - m_bounds[0, 2])  # match z-min ("floor")

    T = trimesh.transformations.translation_matrix([float(dxy[0]), float(dxy[1]), dz])
    mesh.apply_transform(T)

    return mesh

def render_photo_open3d(mesh,
                        yaw_deg=30.0,
                        pitch_deg=20.0,
                        r=5.0,
                        resolution=(1024, 1024),
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
    W, H = map(int, resolution)
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    scene = renderer.scene

    # Background (RGBA if supported; RGB fallback)
    try:
        scene.set_background(bg_color)
    except TypeError:
        scene.set_background(bg_color[:3])

    # Material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = base_color
    if hasattr(mat, "base_roughness"): mat.base_roughness = 0.6
    if hasattr(mat, "base_metallic"):  mat.base_metallic = 0.0
    scene.add_geometry("mesh", o3d_mesh, mat)

    # ---- Camera ----
    aspect = W / float(H)
    bbox_extent = float(np.linalg.norm(np.asarray(mesh.bounding_box.extents, float)))
    near = max(1e-3, 0.01 * max(1.0, r))
    far  = r + 4.0 * max(1.0, bbox_extent) + 10.0 * near
    scene.camera.set_projection(
        fov_deg, aspect, near, far,
        o3d.visualization.rendering.Camera.FovType.Vertical
    )
    scene.camera.look_at(center, eye, up)

    # ---- Lighting (safe across versions) ----
    target = scene if not hasattr(scene, "scene") else scene.scene
    if hasattr(target, "set_ambient_light"):
        try: target.set_ambient_light([0.25, 0.25, 0.25])
        except Exception: pass
    if hasattr(target, "add_directional_light"):
        try: target.add_directional_light("sun", [-0.5, -0.5, -1.0], [1, 1, 1], 90000.0, True)
        except Exception: pass

    # ---- Render ----
    o3d_img = renderer.render_to_image()

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

class Optimize3DTrainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="optimize-3d",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}}
        )

        set_seed(config.seed, device_specific=True)

        self.pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        self.pipeline.to(self.device)

        self.dc_estimator = DragCoefficientEstimator()
        self.ref_mesh = trimesh.load(config.ref_mesh_path)
        self.prompt = getattr(config, "prompt", "A car.")

        self._initialize_structure_distribution()
        self._initialize_slat_distribution()

        self.value_model = ValueModel(dimension=self.slat_dimension)
        self.value_model.to(self.device)

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


    def run(self):
        if self.config.eval_freq > 0:
            self.evaluate(step=0)

        for step in range(1, self.config.optimization_steps+1):
            self.train_step(step)

            if self.config.eval_freq > 0 and (step) % self.config.eval_freq == 0:
                self.evaluate(step)

    def train_step(self, step: int) -> None:
        noise, noise_projected = get_noise(self.mu, self.sigma, self.config.batch_size, self.device)

        noise_slat, noise_projected_slat = get_noise(
            self.mu_slat,
            self.sigma_slat,
            self.config.batch_size,
            self.device,
        )

        meshes, slats = self.generate(struct_noise=noise_projected, slat_noise=noise_projected_slat)
        objective_values = self.dc_estimator(meshes)
        objective_values = torch.from_numpy(objective_values).to(self.device)

        self.accelerator.wait_for_everyone()
        total_batch_size = self.config.batch_size * self.accelerator.num_processes
        gathered_objective_values = self.accelerator.gather(objective_values)
        assert total_batch_size == len(gathered_objective_values)

        _noise = einops.rearrange(noise, "T B D -> B T D")
        gathered_noise = einops.rearrange(self.accelerator.gather(_noise), "B T D -> T B D")

        all_objective_values = None
        if self.accelerator.is_main_process:
            all_objective_values = torch.zeros((self.config.num_inference_steps, total_batch_size), device=self.device,)
            all_objective_values[:, :] = gathered_objective_values[None, :]
            self.mu, self.sigma = update_parameters(self.mu, self.sigma, gathered_noise, all_objective_values)
        self.mu = accelerate.utils.broadcast(self.mu)
        self.sigma = accelerate.utils.broadcast(self.sigma)

        _noise_slat = einops.rearrange(noise_slat, "T B D -> B T D")
        gathered_noise_slat = einops.rearrange(self.accelerator.gather(_noise_slat), "B T D -> T B D")
        if self.accelerator.is_main_process:
            all_objective_values = torch.zeros((self.config.num_inference_steps, total_batch_size), device=self.device,)
            all_objective_values[:, :] = gathered_objective_values[None, :]
            self.mu_slat, self.sigma_slat = update_parameters(self.mu_slat, self.sigma_slat, gathered_noise_slat, all_objective_values)
        self.mu_slat = accelerate.utils.broadcast(self.mu_slat)
        self.sigma_slat = accelerate.utils.broadcast(self.sigma_slat)

        self.log_meshes(meshes, slats, objective_values, step=step, stage="train")
        self.log_objective_metrics(objective_values,step=step,stage="train")

    def evaluate(self, step: int) -> None:
        prompts_idx = self.config.batch_size * self.accelerator.process_index + torch.arange(
            self.config.batch_size, device=self.device
        )
        eval_noise = torch.load("eval_noise/struct_tensor.pt", map_location=self.device)[:, prompts_idx, :]
        struct_prior = eval_noise[0]
        struct_intermidiate = eval_noise[1:]

        eval_noise_slat = torch.load("eval_noise/slat_tensor.pt", map_location=self.device)[:, prompts_idx, :]
        slat_prior = eval_noise_slat[0]
        slat_intermidiate = eval_noise_slat[1:]

        _, noise_projected = get_noise(
            self.mu,
            self.sigma,
            self.config.batch_size,
            self.device,
            base_noise=struct_intermidiate,
        )
        _, noise_projected_slat = get_noise(
            self.mu_slat,
            self.sigma_slat,
            self.config.batch_size,
            self.device,
            base_noise=slat_intermidiate,
        )

        meshes, slats = self.generate(
            struct_prior=struct_prior,
            struct_noise=noise_projected,
            slat_prior=slat_prior,
            slat_noise=noise_projected_slat
        )

        objective_values = self.dc_estimator(meshes)
        objective_values = torch.from_numpy(objective_values).to(self.device)
        self.log_meshes(meshes, slats, objective_values, step, stage="eval")

        self.log_objective_metrics(objective_values, step=step, stage="eval")
        self.save_parameters(step)

    def generate(
        self,
        struct_prior: Optional[torch.Tensor] = None,
        struct_noise: torch.Tensor = None,
        slat_prior: Optional[torch.Tensor] = None,
        slat_noise: torch.Tensor = None,
    ) -> Tuple[List[trimesh.Trimesh], List[torch.Tensor]]:
        meshes: List[trimesh.Trimesh] = []
        slats: List[torch.Tensor] = []

        struct_noise = self._unflatten_structure(struct_noise)
        slat_noise = self._unflatten_slat(slat_noise) if slat_noise is not None else None

        for i in range(self.config.batch_size):
            sparse_structure_sampler_params = {
                "steps": self.config.num_inference_steps,
                "noise_level": 0.7,
                **({"prior_noise": struct_prior[i:i + 1, :]} if struct_prior is not None else {}),
                "intermediate_noise": struct_noise[:, i:i + 1, :],
            }

            slat_sampler_params = {
                "steps": self.config.num_inference_steps,
                "noise_level": 0.7,
                **({"prior_noise": slat_prior[i:i + 1, :]} if slat_prior is not None else {}),
                "intermediate_noise": slat_noise[:, i:i + 1, :],
            }

            cond = self.pipeline.get_cond([self.prompt])
            coords = self.pipeline.sample_sparse_structure(cond, 1, sparse_structure_sampler_params)
            slat = self.pipeline.sample_slat(cond, coords, slat_sampler_params)
            outputs = self.decode_slat(slat, ["mesh"])
            mesh = to_trimesh(outputs["mesh"][0])
            mesh = post_process_mesh(mesh, self.ref_mesh)
            
            meshes.append(mesh)
            slats.append(slat)

        return meshes, slats

    def save_parameters(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            return

        parameters = {
            "mu": self.mu,
            "sigma": self.sigma,
            "mu_slat": self.mu_slat,
            "sigma_slat": self.sigma_slat,
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
        self.accelerator.wait_for_everyone()
        gather_meshes = self.accelerator.gather_for_metrics(meshes)
        gather_slats = self.accelerator.gather_for_metrics(slats)
        gather_objective_values = self.accelerator.gather(objective_values)

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
                    render_photo_open3d(mesh, **view_param),
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
        self.accelerator.wait_for_everyone()
        gathered_objective_values = self.accelerator.gather(objective_values)

        prefix = "" if stage == "train" else f"{stage}_"
        total_batch_size = self.config.batch_size * self.accelerator.num_processes
        metrics = {
            f"{prefix}objective_values_mean": gathered_objective_values.mean().item(),
            f"{prefix}objective_values_std": gathered_objective_values.std().item(),
            "objective_evaluations": total_batch_size * step,
            "mu_norm": self.mu.norm().item(),
        }

        self.accelerator.log(metrics, step=step)

    def _initialize_structure_distribution(self) -> None:
        structure_model = self.pipeline.models["sparse_structure_flow_model"]
        self.structure_resolution = structure_model.resolution
        self.structure_channels = structure_model.in_channels
        self.structure_dimension = self.structure_channels * self.structure_resolution ** 3
        steps = self.config.num_inference_steps
        self.mu = torch.zeros((steps, self.structure_dimension), device=self.device)
        self.sigma = torch.ones((steps, self.structure_dimension), device=self.device)

    def _initialize_slat_distribution(self) -> None:
        self.slat_length = 30000
        slat_model = self.pipeline.models.get("slat_flow_model")
        self.slat_channels = slat_model.in_channels
        self.slat_dimension = self.slat_channels * self.slat_length
        steps = self.config.num_inference_steps
        self.mu_slat = torch.zeros((steps, self.slat_dimension), device=self.device)
        self.sigma_slat = torch.ones((steps, self.slat_dimension), device=self.device)

    def _unflatten_structure(self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            tensor,
            "... (c r1 r2 r3) -> ... c r1 r2 r3",
            c=self.structure_channels,
            r1=self.structure_resolution,
            r2=self.structure_resolution,
            r3=self.structure_resolution,
        )

    def _unflatten_slat(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return einops.rearrange(
            tensor,
            "... (c l) -> ... c l",
            c=self.slat_channels,
            l=self.slat_length,
        )


def main(_):
    trainer = Optimize3DTrainer(FLAGS.config)
    trainer.run()


if __name__ == "__main__":
    app.run(main)

