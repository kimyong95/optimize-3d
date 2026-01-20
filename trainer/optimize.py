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
from torch.utils.data import DataLoader
from ml_collections import config_flags
from PIL import Image
from rewards import ObjectiveEvaluator
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from value_model import ValueModel
from utils import collect_calls, to_trimesh, post_process_mesh
from trainer.base_trainer import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize.py", "Training configuration.")


def update_parameters(mu, sigma, noise, objective_values, lr=1.0):
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

    lr_mu = lr
    lr_sigma = lr / math.sqrt(D_dim)

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


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self._init_parameters()

        self.value_model = ValueModel(dimension=self.structure_dimension)
        self.value_model.to(self.device)

        self.batch_size_per_device = min(config.max_batch_size_per_device, config.total_num_samples // self.accelerator.num_processes)

        # assume train and eval use the same total number of samples and batch size
        self.dataloader = DataLoader(range(config.total_num_samples), batch_size=self.batch_size_per_device)
        self.dataloader = self.accelerator.prepare(self.dataloader)

    def run(self):
        if self.config.eval_freq > 0:
            self.evaluate(step=0)

        for step in range(1, self.config.optimization_steps+1):
            self.train_step(step)

            if self.config.eval_freq > 0 and (step) % self.config.eval_freq == 0:
                self.evaluate(step)

    def train_step(self, step: int) -> None:

        all_noise = []
        all_objective_values = []
        for batch in self.dataloader:
            batch_size = batch.shape[0]

            noise, noise_projected = get_noise(self.mu, self.sigma, self.batch_size_per_device, self.device)
            meshes, slats, pred_data_trajectory = self.generate(batch_size=batch_size, intermediate_noise=noise_projected)
            objective_values = self.objective_evaluator(meshes)
            objective_values = objective_values.to(device=self.device, dtype=torch.float32)
            all_noise.append(noise)
            all_objective_values.append(objective_values)

        all_noise = torch.cat(all_noise, dim=1)
        all_objective_values = torch.cat(all_objective_values, dim=0)

        gathered_objective_values = self.accelerator.gather(all_objective_values)
        gathered_objective_values_mean = gathered_objective_values.mean(dim=-1)
        assert self.config.total_num_samples == len(gathered_objective_values)

        gathered_noise = einops.rearrange(self.accelerator.gather(einops.rearrange(all_noise, "T B D -> B T D")), "B T D -> T B D")

        _pred_data_trajectory = einops.rearrange(pred_data_trajectory, "T B ... -> B T (...)")
        gathered_pred_data_trajectory = einops.rearrange(self.accelerator.gather(_pred_data_trajectory), "B T D -> T B D")

        traj_objective_values = None
        if self.accelerator.is_main_process:
            traj_objective_values = torch.zeros((self.config.num_inference_steps, self.config.total_num_samples), device=self.device,)
            
            self.value_model.add_model_data(
                x = gathered_pred_data_trajectory[-1],
                y = gathered_objective_values_mean,
            )

            for t in range(self.config.num_inference_steps - 1):
                y, _ = self.value_model.predict(gathered_pred_data_trajectory[t])
                traj_objective_values[t] = y
            self.mu, self.sigma = update_parameters(self.mu, self.sigma, gathered_noise, traj_objective_values, lr=self.config.lr)
        self.mu = accelerate.utils.broadcast(self.mu)
        self.sigma = accelerate.utils.broadcast(self.sigma)

        self.log_objective_metrics(gathered_objective_values, objective_evaluations=self.config.total_num_samples * step, stage="train")

    def evaluate(self, step: int) -> None:
        eval_noise = torch.load("eval-noise/struct-tensor.pt", map_location=self.device)
        prior_noise = eval_noise[0]
        intermidiate_noise = eval_noise[1:]

        all_meshes = []
        all_slats = []
        all_objective_values = []
        for batch in self.dataloader:
            batch_size = batch.shape[0]

            _, noise_projected = get_noise(
                self.mu,
                self.sigma,
                batch_size,
                self.device,
                base_noise=intermidiate_noise[:, batch, :],
            )

            meshes, slats, _ = self.generate(
                batch_size=batch_size,
                prior_noise=prior_noise[batch],
                intermediate_noise=noise_projected,
            )

            objective_values = self.objective_evaluator(meshes)
            objective_values = objective_values.to(self.device)

            all_meshes.extend(meshes)
            all_slats.extend(slats)
            all_objective_values.append(objective_values)
        all_objective_values = torch.cat(all_objective_values, dim=0)
        
        objective_evaluations = self.config.total_num_samples * step
        self.log_objective_metrics(all_objective_values, objective_evaluations=objective_evaluations, stage="eval")
        self.save_parameters(step)
        self.accelerator.wait_for_everyone()

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int,
        prior_noise: Optional[torch.Tensor] = None,
        intermediate_noise: torch.Tensor = None,
    ) -> Tuple[List[trimesh.Trimesh], List[torch.Tensor]]:
        meshes: List[trimesh.Trimesh] = []
        slats: List[torch.Tensor] = []
        pred_data_trajectory: List[torch.Tensor] = []
        intermediate_noise = self._unflatten_structure(intermediate_noise)


        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": self.config.noise_level,
            **({"prior_noise": prior_noise} if prior_noise is not None else {}),
            "intermediate_noise": intermediate_noise,
        }

        cond = self.pipeline.get_cond([self.prompt]*batch_size)
        with collect_calls(self.pipeline.sparse_structure_sampler.sample) as collected_data:            
            coords = self.pipeline.sample_sparse_structure(cond, batch_size, sparse_structure_sampler_params)
        pred_data_trajectory = torch.stack(collected_data[0]["return"]["pred_x_0"][1:] + [collected_data[0]["return"]["samples"]])

        slats = self.pipeline.sample_slat(cond, coords)

        meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
        meshes = [ to_trimesh(mesh) for mesh in meshes ]
        meshes = [ post_process_mesh(mesh) for mesh in meshes ]

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

if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
