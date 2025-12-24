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
from tqdm import tqdm
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
from base_trainer import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/dno.py", "Training configuration.")



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


class Trainer(BaseTrainer):
    def __init__(self, config):

        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, "Only supports single GPU"

        super().__init__(config)

        [ model[1].requires_grad_(False) for model in self.pipeline.models.items() ] # disable all gradients
        self.pipeline.sample_sparse_structure = self.sample_sparse_structure_dno.__get__(self.pipeline) # replace the original method
        
        structure_model = self.pipeline.models["sparse_structure_flow_model"]
        self.structure_resolution = structure_model.resolution
        self.structure_channels = structure_model.in_channels

        self.pipeline.models["sparse_structure_flow_model"] = enable_gradient_checkpointing(self.pipeline.models["sparse_structure_flow_model"])


    @torch.no_grad()
    def run(self) -> None:
        all_meshes = []
        all_slats = []
        all_objective_values = []
        eval_noise = torch.load("eval-noise/struct-tensor.pt", map_location=self.device)

        for sample_i in tqdm(range(self.config.total_num_samples), desc="Optimizing samples", disable=not self.accelerator.is_main_process, position=0):

            ref_noise = eval_noise[:,sample_i].unsqueeze(1).clone().to(self.device)
            ref_noise = list(ref_noise)
            [x.requires_grad_(True) for x in ref_noise]
            optimizer = torch.optim.AdamW(ref_noise, lr=0.01, weight_decay=0.0)

            meshes = []
            slats = []
            objective_values = []

            try: # try optimize sample_i

                # +1 because we want to log the ref_images after last noise update
                for optimization_i in tqdm(range(self.config.optimization_steps), desc="Optimization steps", leave=False, disable=not self.accelerator.is_main_process, position=1):
                    
                    # ------------- reference ------------- #
                    sparse_structure_sampler_params = {
                        "steps": self.config.num_inference_steps,
                        "noise_level": 0.7,
                        "prior_noise": ref_noise[0],
                        "intermediate_noise": ref_noise[1:],
                    }
                    cond = self.pipeline.get_cond([self.prompt])
                    with torch.enable_grad():
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
                    noise = noise + torch.randn_like(noise) * 0.01
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

            except Exception as e: # fail to optimize sample_i
                print(f"Sample {sample_i} failed at optimization step {optimization_i} with error: {e}")
            
            else: # successfully optimize sample_i
                all_meshes.append(meshes)
                all_slats.append(slats)
                all_objective_values.append(torch.cat(objective_values))
            
            torch.cuda.empty_cache()

        # flip: (B, optimization_steps) -> (optimization_steps, B)
        all_objective_values = torch.stack(all_objective_values).T
        all_steps_meshes = list(map(list, zip(*all_meshes)))
        all_steps_slats = list(map(list, zip(*all_slats)))

        # log trajectory objective values
        for i in range(self.config.optimization_steps):
            objective_evaluations = self.config.total_num_samples * (self.config.batch_size + 1) * (i + 1)
            self.log_objective_metrics(all_objective_values[i], objective_evaluations=objective_evaluations)
            self.log_meshes(
                all_steps_meshes[i],
                all_steps_slats[i],
                all_objective_values[i],
                objective_evaluations=objective_evaluations,
            )

        self.accelerator.log({"successful_samples": len(all_meshes)})
        self.accelerator.log({"failed_samples": self.config.total_num_samples - len(all_meshes)})
        

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



if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
