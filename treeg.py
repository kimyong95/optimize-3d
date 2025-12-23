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
config_flags.DEFINE_config_file("config", "config/treeg.py", "Training configuration.")


class Trainer(BaseTrainer):
    def __init__(self, config):

        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, "Only supports single GPU"

        super().__init__(config)

        self.pipeline.sparse_structure_sampler.sample_once_original = self.pipeline.sparse_structure_sampler.sample_once
        self.pipeline.sparse_structure_sampler.sample_once = self.sample_once_treeg.__get__(self.pipeline.sparse_structure_sampler)

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
                self.log_objective_metrics(objective_values_t, objective_evaluations=self.config.total_num_samples * self.config.expansion_size * (t+1))
        
        self.log_meshes(all_meshes,all_slats,all_log_trajectory_obj_values[:, -1],step=self.config.num_inference_steps-1,stage="eval")

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



if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
