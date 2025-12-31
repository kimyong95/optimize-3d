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
from base_trainer import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/dsearch.py", "Training configuration.")

class Trainer(BaseTrainer):
    def __init__(self, config):

        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, "Only supports single GPU"

        super().__init__(config)

        self.pipeline.sparse_structure_sampler.sample_once_original = self.pipeline.sparse_structure_sampler.sample_once
        self.pipeline.sparse_structure_sampler.sample_once = self.sample_once_dsearch.__get__(self.pipeline.sparse_structure_sampler)

        # b_t in the paper
        self.batch_size_t = [ int( config.init_batch_size * (config.final_batch_size/config.init_batch_size)**(t/config.num_inference_steps) ) for t in range(config.num_inference_steps) ]

        # w_t in the paper
        self.expansion_size_t = [ int(config.evaluation_budget // b_t) for b_t in self.batch_size_t ]

        objective_evaluations = [b*e for b,e in zip(self.batch_size_t, self.expansion_size_t)]
        self.objective_evaluations = torch.tensor(objective_evaluations).cumsum(dim=0)
        self.batch_size_t.append(config.final_batch_size)
        
    @torch.inference_mode()
    def run(self) -> None:

        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": 0.7,
            "external_self": self,
        }

        cond = self.pipeline.get_cond([self.prompt]*self.config.init_batch_size)
        coords = self.pipeline.sample_sparse_structure(cond, self.config.init_batch_size, sparse_structure_sampler_params)
        
        cond = self.pipeline.get_cond([self.prompt]*self.config.final_batch_size)
        meshes, slats = self.generate_meshes_from_coords(cond, coords)

        objective_values = self.objective_evaluator(meshes).to(self.device)
        self.log_objective_metrics(objective_values, objective_evaluations=int(self.objective_evaluations[-1]), stage="final")
        self.log_meshes(meshes, slats, objective_values, objective_evaluations=int(self.objective_evaluations[-1]), stage="final")

    @staticmethod
    @torch.no_grad()
    def sample_once_dsearch(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        noise_level: float = 0.0,
        noise: Optional[torch.Tensor] = None, # this will be ignored
        # ----------- dsearch args ----------- #
        external_self = None,
        # ----------- dsearch args ----------- #
        **kwargs
    ):
        """
        To replace the original sample_once with dsearch sampling
        """
        batch_size = x_t.shape[0]
        rescale_t = external_self.pipeline.sparse_structure_sampler_params["rescale_t"]
        t_seq = np.linspace(1, 0, external_self.config.num_inference_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_idx = np.argwhere(t_seq == t).item()
        t_prev_idx = t_idx + 1
        t_prev_prev_idx = t_idx + 2
        cond_one = cond[0].unsqueeze(0)
        cond_dict_one = external_self.pipeline.get_cond([external_self.prompt])
        assert torch.allclose(cond, cond_one.expand_as(cond))

        # ----------- original code ----------- #
        cond_batch = einops.repeat(cond_one, "1 ... -> b ...", b=batch_size)
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond_batch, **kwargs)
        dt = t_prev - t

        std_dev_t = np.sqrt(t / (1 - np.where(t == 1, t_prev, t)))*noise_level
        x_prev_mean = x_t*(1+std_dev_t**2/(2*t)*dt) + pred_v*(1+std_dev_t**2*(1-t)/(2*t))*dt
        
        # ------------ dsearch code ------------- #
        expansion_size = external_self.expansion_size_t[t_idx]
        x_prev_candidates = []
        x_prev_candidates_obj_values = []
        
        for i, x_prev_mean_i in enumerate(x_prev_mean):
            # sample
            noise_i = torch.randn( (expansion_size,) + x_prev_mean_i.shape, device=x_prev_mean_i.device)
            x_prev_i = x_prev_mean_i + std_dev_t * np.sqrt(-1*dt) * noise_i
            x_prev_candidates.append(x_prev_i)
            
            # determinisitcally sample once, decode and evaluate
            objective_values = torch.full((expansion_size, external_self.objective_evaluator.num_objectives), float('inf'), device=x_t.device)
            for j, x_prev_ij in enumerate(x_prev_i):
                try:
                    pred_sample_ij = self.sample_once_original(model, x_prev_ij.unsqueeze(0), t_prev, t_seq[t_prev_prev_idx], cond_one, noise_level=0.0, **kwargs).pred_x_0 if t_prev_prev_idx < len(t_seq) else x_prev_ij.unsqueeze(0)
                    coords = torch.argwhere(external_self.pipeline.models['sparse_structure_decoder'](pred_sample_ij)>0)[:, [0, 2, 3, 4]].int()
                    meshes, slats = external_self.generate_meshes_from_coords(cond_dict_one, coords)
                    objective_values[j] = external_self.objective_evaluator(meshes).to(x_t.device)
                except Exception as e:
                    print(f"Exception {e} when decoding at {t_idx}-th timestep, assign inf objective values")
            x_prev_candidates_obj_values.append(objective_values)

        # flatten
        x_prev_candidates = torch.stack(x_prev_candidates, dim=0)
        x_prev_candidates_obj_values = torch.stack(x_prev_candidates_obj_values, dim=0) # (batch_size, expansion_size, num_objectives)

        # instance-wise best
        best_indices = x_prev_candidates_obj_values.mean(dim=-1).argmin(dim=1)
        x_prev_candidates = x_prev_candidates[torch.arange(len(x_prev_candidates)), best_indices]
        x_prev_candidates_obj_values = x_prev_candidates_obj_values[torch.arange(len(x_prev_candidates_obj_values)), best_indices]

        # global selection
        next_batch_size = external_self.batch_size_t[t_prev_idx]
        next_indices = x_prev_candidates_obj_values.mean(dim=-1).topk(next_batch_size, largest=False).indices
        x_prev = x_prev_candidates[next_indices]
        x_prev_candidates_obj_values = x_prev_candidates_obj_values[next_indices]
        pred_x_0 = pred_x_0[next_indices]

        # log
        if torch.isfinite(x_prev_candidates_obj_values).all():
            external_self.log_objective_metrics(x_prev_candidates_obj_values, objective_evaluations=external_self.objective_evaluations[t_idx])

        return edict({"pred_x_prev": x_prev, "pred_x_0": pred_x_0})


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
