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
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import trimesh
from typing import List, Optional, Tuple, Any, Union
import itertools
from peft import LoraConfig, get_peft_model
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
from utils import collect_calls
from tqdm import tqdm
import trellis.modules.sparse as sp
from utils import collect_calls, to_trimesh, post_process_mesh

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/grpo.py", "Training configuration.")

def batches_dict(data, batch_size):
    n = len(next(iter(data.values())))
    for i in range(0, n, batch_size):
        yield {k: v[i:i+batch_size] for k, v in data.items()}

def concat(data: Union[List[torch.Tensor], List[List]]):
    if isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)
    elif isinstance(data[0], list):
        return sum(data, [])
    elif isinstance(data[0], sp.SparseTensor):
        return sp.sparse_cat(data, dim=0)
    else:
        raise ValueError(f"Unsupported data type: {type(data[0])}")


class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
        self.accelerator.init_trackers(
            project_name="optimize-3d",
            init_kwargs={"wandb": {"name": config.run_name, "config": config.to_dict()}}
        )

        set_seed(config.seed, device_specific=True)

        self.pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        self.pipeline.to(self.device)
        [ model[1].requires_grad_(False) for model in self.pipeline.models.items() ] # disable all gradients
        target_modules = [
            "to_qkv",
            "to_q",
            "to_kv",
            "to_out",
        ]
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.pipeline.models['sparse_structure_flow_model'] = get_peft_model(self.pipeline.models['sparse_structure_flow_model'], lora_config)
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.pipeline.models['sparse_structure_flow_model'].parameters()))
        self.optimizer = torch.optim.AdamW(trainable_parameters,lr=self.config.learning_rate,)

        # to handle if the sampling_max_batch_size_per_device * num_processes > training_samples_per_epoch
        self.sampling_batch_size_per_device = min(config.sampling_max_batch_size_per_device, config.training_samples_per_epoch // self.accelerator.num_processes)
        train_dataloader = DataLoader(torch.arange(config.training_samples_per_epoch), batch_size=self.sampling_batch_size_per_device)

        # to handle if the eval_max_batch_size_per_device * num_processes > eval_samples
        self.eval_batch_size_per_device = min(config.eval_max_batch_size_per_device, config.eval_samples // self.accelerator.num_processes)
        eval_dataloader = DataLoader(torch.arange(config.eval_samples), batch_size=self.eval_batch_size_per_device)

        self.accelerator.gradient_accumulation_steps = self.config.num_inference_steps * (self.config.training_effective_batch_size // (self.config.training_max_batch_size_per_device * self.accelerator.num_processes))
        
        self.pipeline.models['sparse_structure_flow_model'], self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(self.pipeline.models['sparse_structure_flow_model'], self.optimizer, train_dataloader, eval_dataloader)

        self.objective_evaluator = ObjectiveEvaluator(objective=config.objective, port=config.reward_server_port)
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

    def run(self):
        if self.config.eval_freq > 0:
            self.evaluation_step(epoch=0, noisy=True)
            self.evaluation_step(epoch=0, noisy=False)
            
        
        for epoch in tqdm(range(1, self.config.epoches+1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            
            training_data, training_kwargs = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data, training_kwargs=training_kwargs)

            if self.config.eval_freq > 0 and epoch % self.config.eval_freq == 0:
                self.evaluation_step(epoch=epoch, noisy=True)
                self.evaluation_step(epoch=epoch, noisy=False)
                

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.models['sparse_structure_flow_model'].eval()
        training_data = []
        training_kwargs = []
        all_meshes = []
        all_slats = []
        for data_ids in tqdm(self.train_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            batch_size = len(data_ids)

            cond = self.pipeline.get_cond([self.prompt]*batch_size)
            with collect_calls(self.pipeline.sparse_structure_sampler.sample_once, arg_names=["x_t", "t", "t_prev", "cond", "neg_cond", "cfg_strength", "cfg_interval"]) as collected_data:            
                coords = self.pipeline.sample_sparse_structure(cond, batch_size, { "noise_level": 0.7 })

            slats = self.pipeline.sample_slat(cond, coords)
            meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
            meshes = [ to_trimesh(mesh) for mesh in meshes ]
            meshes = [ post_process_mesh(mesh) for mesh in meshes ]

            objective_values = self.objective_evaluator(meshes)
            objective_values = torch.from_numpy(objective_values).to(self.device)

            # each element is len of batch_size
            training_data.append({
                    "cond": torch.stack([d["args"]["cond"] for d in collected_data],dim=1),
                    "sample": torch.stack([d["args"]["x_t"] for d in collected_data],dim=1),
                    "prev_sample": torch.stack([d["return"]["pred_x_prev"] for d in collected_data],dim=1),
                    "prev_sample_mean": torch.stack([d["return"]["pred_x_prev_mean"] for d in collected_data],dim=1),
                    "objective_values": objective_values,
            })
            
            training_kwargs_names = ["neg_cond", "cfg_strength", "cfg_interval", "t", "t_prev"]
            training_kwargs.append([{k: d["args"][k] for k in training_kwargs_names if k in d["args"]} for d in collected_data])

            all_meshes.append(meshes)
            all_slats.append(slats)
            del collected_data
        
        training_data = {
            key: concat([batch[key] for batch in training_data])
            for key in training_data[0].keys()
        }

        assert all(x == training_kwargs[0] for x in training_kwargs)
        training_kwargs = training_kwargs[0]

        objective_values = training_data["objective_values"]
        gathered_objective_values = self.accelerator.gather(objective_values)
        gathered_rewards = - gathered_objective_values
        gathered_advantages = ( gathered_rewards - gathered_rewards.mean() ) / (gathered_rewards.std(unbiased=False) + 1e-4) # unbiased=False to match np.std() in the official source code

        training_data["advantages"] = einops.rearrange(gathered_advantages,'(process batch) -> process batch',process=self.accelerator.num_processes)[self.accelerator.process_index]

        non_zero_mask = training_data["advantages"] != 0
        training_data = {
            key: value[non_zero_mask] if isinstance(value, torch.Tensor) else list(itertools.compress(value, non_zero_mask))
            for key, value in training_data.items()
        }

        self.log_objective_metrics(objective_values=objective_values, step=epoch, stage="train")

        return training_data, training_kwargs

    def training_step(self, epoch, training_data, training_kwargs):
        self.pipeline.models['sparse_structure_flow_model'].train()

        training_batches = list(batches_dict(training_data, self.config.training_max_batch_size_per_device))
        
        for training_batch in tqdm(training_batches, desc="Training Batches", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            batch_size = len(training_batch["sample"])

            for i in tqdm(range(self.config.num_inference_steps), desc="Timesteps", position=2, leave=False, disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(self.pipeline.models['sparse_structure_flow_model']):

                    training_kwargs_i = training_kwargs[i]
                    t = training_kwargs_i["t"]
                    t_prev = training_kwargs_i["t_prev"]

                    with torch.enable_grad(), self.accelerator.autocast():
                        prev_sample_mean = self.pipeline.sparse_structure_sampler.sample_once(self.pipeline.models['sparse_structure_flow_model'], training_batch["sample"][:,i], cond=training_batch["cond"][:,i], noise_level=0.7, **training_kwargs_i).pred_x_prev_mean
                        
                    with self.accelerator.unwrap_model(self.pipeline.models['sparse_structure_flow_model']).disable_adapter(), torch.no_grad(), self.accelerator.autocast():
                        prev_sample_mean_ref = self.pipeline.sparse_structure_sampler.sample_once(self.pipeline.models['sparse_structure_flow_model'], training_batch["sample"][:,i], cond=training_batch["cond"][:,i], noise_level=0.7, **training_kwargs_i).pred_x_prev_mean

                    curr_log_prob = self.pipeline.sparse_structure_sampler.calculate_log_prob(pred_x_prev_mean=prev_sample_mean, pred_x_prev=training_batch["prev_sample"][:,i], t=t, t_prev=t_prev, noise_level=0.7)

                    old_log_prob = self.pipeline.sparse_structure_sampler.calculate_log_prob(pred_x_prev_mean=training_batch["prev_sample_mean"][:,i], pred_x_prev=training_batch["prev_sample"][:,i], t=t, t_prev=t_prev, noise_level=0.7)

                    advantages = torch.clamp(
                        training_batch["advantages"],
                        -self.config.adv_clip_max,
                        self.config.adv_clip_max,
                    )
                    ratio = torch.exp(curr_log_prob - old_log_prob)
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    std_dev_t = self.pipeline.sparse_structure_sampler.get_std_dev_t(t, t_prev)
                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                    kl_loss = torch.mean(kl_loss)

                    loss = policy_loss + self.config.kl_beta * kl_loss

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.pipeline.models['sparse_structure_flow_model'].parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluation_step(self, epoch, noisy=False):
        
        self.pipeline.models['sparse_structure_flow_model'].eval()
        all_objective_values = []
        all_meshes = []
        all_slats = []
        for data_ids in tqdm(self.eval_dataloader, desc="Evaluation", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            batch_size = len(data_ids)
            eval_noise = torch.load("eval_noise/struct_tensor.pt", map_location=self.device)[:, data_ids, :]

            if noisy:
                sparse_structure_sampler_params = {
                    "steps": self.config.num_inference_steps,
                    "noise_level": 0.7,
                    "prior_noise": eval_noise[0],
                    "intermediate_noise": eval_noise[1:],
                }
            else:
                sparse_structure_sampler_params = {
                    "steps": self.config.num_inference_steps,
                    "noise_level": 0.0,
                    "prior_noise": eval_noise[0],
                }
            cond = self.pipeline.get_cond([self.prompt]*batch_size)
            coords = self.pipeline.sample_sparse_structure(cond, batch_size, sparse_structure_sampler_params)
            slats = self.pipeline.sample_slat(cond, coords)

            meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
            meshes = [ to_trimesh(mesh) for mesh in meshes ]
            meshes = [ post_process_mesh(mesh) for mesh in meshes ]
            objective_values = self.objective_evaluator(meshes)
            objective_values = torch.from_numpy(objective_values).to(self.device)

            all_objective_values.append(objective_values)
            all_meshes.extend(meshes)
            all_slats.extend(slats)

        all_objective_values = torch.cat(all_objective_values, dim=0)

        stage = "eval_noisy" if noisy else "eval"

        self.log_meshes(all_meshes, all_slats, all_objective_values, epoch, stage=stage)
        self.log_objective_metrics(all_objective_values, step=epoch, stage=stage)
        self.accelerator.wait_for_everyone()

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
        stage: str,
    ) -> None:
        gathered_objective_values = self.accelerator.gather(objective_values)
        self.accelerator.wait_for_everyone()

        prefix = {
            "train": "",
            "eval": "eval_",
            "eval_noisy": "eval_noisy_",
        }
        metrics = {
            prefix[stage] + "objective_values_mean": gathered_objective_values.mean().item(),
            prefix[stage] + "objective_values_std": gathered_objective_values.std().item(),
            "objective_evaluations": self.config.training_samples_per_epoch * step,
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
