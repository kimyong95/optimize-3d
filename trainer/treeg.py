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
from trainer.base_trainer import BaseTrainer

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
        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": self.config.noise_level,
            "cfg_strength": self.config.guidance_scale,
            "external_self": self,
        }

        cond = self.pipeline.get_cond([self.prompt]*self.config.batch_size)
        coords = self.pipeline.sample_sparse_structure(cond, self.config.batch_size, sparse_structure_sampler_params)

        meshes, slats = self.generate_meshes_from_coords(cond, coords)

        total_objective_evaluations = self.config.batch_size * self.config.expansion_size * self.config.num_inference_steps
        objective_values = self.objective_evaluator(meshes).to(self.device)
        self.log_objective_metrics(objective_values, objective_evaluations=total_objective_evaluations, stage="final")
        self.log_meshes(meshes, slats, objective_values, objective_evaluations=total_objective_evaluations, stage="final")

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
        cond_one = cond[0].unsqueeze(0)
        cond_dict_one = external_self.pipeline.get_cond([external_self.prompt])
        assert torch.allclose(cond, cond_one.expand_as(cond))
        # ----------- original code ----------- #
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        dt = t_prev - t

        std_dev_t = np.sqrt(t / (1 - np.where(t == 1, t_prev, t)))*noise_level
        x_prev_mean = x_t*(1+std_dev_t**2/(2*t)*dt) + pred_v*(1+std_dev_t**2*(1-t)/(2*t))*dt
        
        # ------------ treeg code ------------- #
        x_prev_candidates = []
        x_prev_candidates_obj_values = []
        x_prev_candidates_meshes = []
        x_prev_candidates_slats = []
        for i, x_prev_mean_i in enumerate(x_prev_mean):
            # sample
            noise_i = torch.randn( (external_self.config.expansion_size,) + x_prev_mean_i.shape, device=x_prev_mean_i.device)
            x_prev_i = x_prev_mean_i + std_dev_t * np.sqrt(-1*dt) * noise_i
            x_prev_candidates.append(x_prev_i)

            # determinisitcally sample once, decode and evaluate
            objective_values = torch.full((external_self.config.expansion_size, external_self.objective_evaluator.num_objectives), float('inf'), device=x_t.device)
            meshes_i = [None] * external_self.config.expansion_size
            slats_i = [None] * external_self.config.expansion_size
            for j, x_prev_ij in enumerate(x_prev_i):
                try:
                    pred_sample_ij = self.sample_once_original(model, x_prev_ij.unsqueeze(0), t_prev, t_seq[t_prev_prev_idx], cond_one, noise_level=0.0, **kwargs).pred_x_0 if t_prev_prev_idx < len(t_seq) else x_prev_ij.unsqueeze(0)
                    coords = torch.argwhere(external_self.pipeline.models['sparse_structure_decoder'](pred_sample_ij)>0)[:, [0, 2, 3, 4]].int()
                    meshes, slats = external_self.generate_meshes_from_coords(cond_dict_one, coords)
                    objective_values[j] = external_self.objective_evaluator(meshes).to(x_t.device)
                    meshes_i[j] = meshes[0]
                    slats_i[j] = slats[0]
                except Exception as e:
                    print(f"Exception {e} when decoding at {t_idx}-th timestep, assign inf objective values")
            x_prev_candidates_obj_values.append(objective_values)
            x_prev_candidates_meshes.extend(meshes_i)
            x_prev_candidates_slats.extend(slats_i)

        # flatten
        x_prev_candidates = torch.cat(x_prev_candidates, dim=0)
        x_prev_candidates_obj_values = torch.cat(x_prev_candidates_obj_values, dim=0)

        # global selection
        next_indices = x_prev_candidates_obj_values.mean(dim=-1).topk(batch_size, largest=False).indices
        x_prev_candidates_obj_values = x_prev_candidates_obj_values[next_indices]
        x_prev = x_prev_candidates[next_indices]
        selected_meshes = [x_prev_candidates_meshes[i] for i in next_indices.tolist()]
        selected_slats = [x_prev_candidates_slats[i] for i in next_indices.tolist()]

        # log
        if torch.isfinite(x_prev_candidates_obj_values).all():
            obj_evals = external_self.config.batch_size * external_self.config.expansion_size * (t_idx + 1)
            external_self.log_objective_metrics(x_prev_candidates_obj_values, objective_evaluations=obj_evals)
            external_self.log_meshes(selected_meshes, selected_slats, x_prev_candidates_obj_values, objective_evaluations=obj_evals)

        return edict({"pred_x_prev": x_prev, "pred_x_0": pred_x_0})



if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
