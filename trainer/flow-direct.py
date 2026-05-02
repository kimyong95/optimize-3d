import math
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple, Any
import accelerate
import sys
import einops
import numpy as np
import open3d as o3d
from functools import partial
import torch
import torch.nn as nn
import trimesh
from easydict import EasyDict as edict
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
import trellis.modules.sparse as sp

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flow-direct.py", "Training configuration.")


@torch.no_grad()
def distribution_shift(X1, Y_target, xt, x1, t):
    # X1: (N, D...)
    # Y_target: (N,)
    # xt: (D...)
    # x1: (D...)

    N, D = X1.shape[0], X1.shape[1:]
    dtype = xt.dtype

    if N < 2:
        return torch.zeros_like(xt)

    X1 = X1.reshape(N, -1).to(torch.float64)
    Y_target = Y_target.to(torch.float64)
    xt = xt.reshape(1, -1).to(torch.float64)
    x1 = x1.reshape(1, -1).to(torch.float64)

    # add virtual data X
    X1 = torch.cat([X1, x1], dim=0)
    Xt = t * X1 + (1-t) * torch.randn_like(X1)

    # logits: -||xt - Xt_j||^2 / (2*(1-t)^2) / D**0.5
    logits_fn = lambda _xt, _Xt: -torch.cdist(_xt, _Xt, p=2).squeeze(0) ** 2 / (2 * (1-t)**2) / math.prod(D)**0.5
    logits = logits_fn(xt, Xt)

    # add virtual label Y
    y1 = (torch.softmax(logits[:-1], dim=0) * Y_target).sum(dim=0)
    Y_target = torch.cat([Y_target, y1.unsqueeze(0)], dim=0)

    p_base = torch.softmax(logits, dim=0)
    p_target = torch.softmax(logits + Y_target, dim=0)

    E_x1_base = torch.einsum('N, ND -> D', p_base, X1)
    E_x1_target = torch.einsum('N, ND -> D', p_target, X1)

    d = (E_x1_target - E_x1_base).reshape(*D).to(dtype)

    return d

class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.batch_size_per_device = min(config.max_batch_size_per_device, config.total_num_samples // self.accelerator.num_processes)

        self._init_shape()
        self.data = {
            "x1": torch.empty(0, *self.latent_shape, device=self.accelerator.device),
            "rewards": torch.empty(0, device=self.accelerator.device),
        }

        _get_model_prediction_extend = partial(Trainer._get_model_prediction_extend, self.pipeline.sparse_structure_sampler, self)
        self._get_model_prediction_original = self.pipeline.sparse_structure_sampler._get_model_prediction
        self.pipeline.sparse_structure_sampler._get_model_prediction = _get_model_prediction_extend

        # assume train and eval use the same total number of samples and batch size
        self.dataloader = DataLoader(range(config.total_num_samples), batch_size=self.batch_size_per_device)
        self.dataloader = self.accelerator.prepare(self.dataloader)

    def increment_data(self, x1, rewards):
        self.data["x1"] = torch.cat([self.data["x1"], x1], dim=0)
        self.data["rewards"] = torch.cat([self.data["rewards"], rewards], dim=0)
        
        torch.cuda.empty_cache()

    @torch.no_grad()
    def distribution_shift(self, xt, x1, t):
        # Input:
        #   xt: (n, D...)
        #   x1: (n, D...)
        #   t: scalar
        # Output:
        #   d: (n, D...)

        X1 = self.data["x1"]
        Y = self.data["rewards"]

        N, D = X1.shape[0], X1.shape[1:]
        n = xt.shape[0]
        B = self.config.total_num_samples
        L = N // B

        if N == 0:
            return torch.zeros_like(xt)

        finite = torch.isfinite(Y)
        Y_finite = Y[finite]
        Y = (Y - Y_finite.mean()) / Y_finite.std().clamp(min=1e-3)

        d = torch.zeros(L, n, *self.latent_shape, device=xt.device, dtype=xt.dtype)
        for l in range(L):
            mask_l = finite[l*B:(l+1)*B]
            X1_l = X1[l*B:(l+1)*B][mask_l]
            Y_l = Y[l*B:(l+1)*B][mask_l]
            if X1_l.shape[0] == 0:
                continue
            for i in range(n):
                d[l, i] = distribution_shift(X1_l, Y_l, xt[i], x1[i], t)
        d = d.sum(dim=0)
        return d

    def run(self):

        for step in range(1, self.config.optimization_steps+1):
            self.sampling_step(step)

    @torch.inference_mode()
    def sampling_step(self, step: int) -> None:

        all_objective_values = []
        all_x1 = []
        all_meshes = []
        all_slats = []
        for batch in self.dataloader:
            batch_size = batch.shape[0]

            sparse_structure_sampler_params = {"steps": self.config.num_inference_steps, "noise_level": self.config.noise_level}

            cond = self.pipeline.get_cond([self.prompt]*batch_size)
            with collect_calls(self.pipeline.sparse_structure_sampler.sample) as collected_data:
                coords = self.pipeline.sample_sparse_structure(cond, batch_size, sparse_structure_sampler_params)
            meshes, slats = self.generate_meshes_from_coords(cond, coords)
            objective_values = self.objective_evaluator(meshes).to(device=self.device, dtype=torch.float32)
            all_objective_values.append(objective_values)
            all_x1.append(collected_data[0]["return"]["samples"])
            all_meshes.extend(meshes)
            all_slats.extend(slats)

        all_objective_values = torch.cat(all_objective_values, dim=0)
        all_x1 = torch.cat(all_x1, dim=0)
        gathered_objective_values = self.accelerator.gather(all_objective_values)
        gathered_rewards = - gathered_objective_values
        gathered_x1 = self.accelerator.gather(all_x1)
        
        assert self.config.total_num_samples == len(gathered_objective_values)
        self.increment_data(gathered_x1, gathered_rewards.squeeze(-1))

        self.log_objective_metrics(all_objective_values, objective_evaluations=self.config.total_num_samples * step)
        self.log_meshes(all_meshes, all_slats, all_objective_values, objective_evaluations=self.config.total_num_samples * step)

    def _get_model_prediction_extend(self, external_self, model, x_t, t, cond=None, **kwargs):
        
        if x_t.shape[1:] != external_self.latent_shape:
            return self._get_model_prediction_original(model, x_t, t, cond, **kwargs)
        
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)

        time_t = 1 - t
        x1 = x_t + pred_v * (1 - time_t)
        d = external_self.distribution_shift(x_t, x1, time_t)
        pred_d = -1 * d # flip sign because pred_v is negative vector field in Eq.2
        pred_v = pred_v + pred_d.to(pred_v.dtype) / (1-time_t)

        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    def _init_shape(self) -> None:
        structure_model = self.pipeline.models["sparse_structure_flow_model"]
        self.structure_resolution = structure_model.resolution
        self.structure_channels = structure_model.in_channels
        self.latent_shape = (self.structure_channels, self.structure_resolution, self.structure_resolution, self.structure_resolution)

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
