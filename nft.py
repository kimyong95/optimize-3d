import math
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple

import contextlib
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

from base_trainer import BaseTrainer

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

@contextlib.contextmanager
def use_adapter(model, adapter_name: str):
    active_adapter = model.active_adapter

    # Switch to the desired adapter
    model.set_adapter(adapter_name)
    try:
        yield model
    finally:
        model.set_adapter(active_adapter)


class Trainer(BaseTrainer):
    def __init__(self, config):

        num_gpus = torch.cuda.device_count()
        assert config.training_samples_per_epoch % num_gpus == 0
        total_training_samples_per_device = config.training_samples_per_epoch // num_gpus
        
        assert config.eval_samples % num_gpus == 0
        total_eval_samples_per_device = config.eval_samples // num_gpus
        
        with config.unlocked():
            config.sampling_batch_size_per_device = min(config.sampling_max_batch_size_per_device, total_training_samples_per_device)
            config.training_batch_size_per_device = min(config.training_max_batch_size_per_device, total_training_samples_per_device)
            config.eval_batch_size_per_device = min(config.eval_max_batch_size_per_device, total_eval_samples_per_device)

        # calculate gradient accumulation steps
        accumuate_num_train_batches = total_training_samples_per_device / config.training_batch_size_per_device / config.gradient_updates_per_epoch
        assert accumuate_num_train_batches.is_integer(), f"The number of gradient accumulation steps per epoch must be an integer, but got {accumuate_num_train_batches}"
        accumuate_num_train_batches = int(accumuate_num_train_batches)
        gradient_accumulation_steps = config.num_inference_steps * accumuate_num_train_batches

        super().__init__(config, accelerator_kwargs={"gradient_accumulation_steps": gradient_accumulation_steps})

        model_vars = vars(self.pipeline.models['sparse_structure_flow_model'])
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
        self.trainable_parameters = list(filter(lambda p: p.requires_grad, self.pipeline.models['sparse_structure_flow_model'].parameters()))
        self.optimizer = torch.optim.AdamW(self.trainable_parameters,lr=self.config.learning_rate)

        self.pipeline.models['sparse_structure_flow_model'].add_adapter("old", lora_config)
        with use_adapter(self.pipeline.models['sparse_structure_flow_model'], "old"):
            self.old_trainable_parameters = list(filter(lambda p: p.requires_grad, self.pipeline.models['sparse_structure_flow_model'].parameters())) # theta-old in the paper
        self.copy_parameters(self.trainable_parameters, self.old_trainable_parameters, ema_decay=0.0) # initialize old parameters same as current parameters

        train_dataloader = DataLoader(torch.arange(config.training_samples_per_epoch), batch_size=config.sampling_batch_size_per_device)
        eval_dataloader = DataLoader(torch.arange(config.eval_samples), batch_size=config.eval_batch_size_per_device)
        
        self.pipeline.models['sparse_structure_flow_model'], self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(self.pipeline.models['sparse_structure_flow_model'], self.optimizer, train_dataloader, eval_dataloader)
        ddp_model = self.pipeline.models['sparse_structure_flow_model']
        [setattr(ddp_model, k, v) for k, v in model_vars.items() if isinstance(v,int)]

    @staticmethod
    def copy_parameters(sources: List[torch.Tensor], destinations: List[torch.Tensor], ema_decay: float = 0.0):
        for src, dest in zip(sources, destinations, strict=True):
            dest.data.copy_(ema_decay * dest.detach().data + (1.0 - ema_decay) * src.detach().clone().data)

    def run(self):
        if self.config.eval_freq > 0:
            self.evaluation_step(epoch=0)
            
        
        for epoch in tqdm(range(1, self.config.epoches+1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)

            if self.config.eval_freq > 0 and epoch % self.config.eval_freq == 0:
                self.evaluation_step(epoch=0)
                

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.models['sparse_structure_flow_model'].eval()
        training_data = []
        all_meshes = []
        all_slats = []
        for data_ids in tqdm(self.train_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            batch_size = len(data_ids)

            cond = self.pipeline.get_cond([self.prompt]*batch_size)
            with collect_calls(self.pipeline.sparse_structure_sampler.sample_once, arg_names=["x_t", "t", "t_prev", "cond", "neg_cond", "cfg_strength", "cfg_interval"]) as collected_data:            
                coords = self.pipeline.sample_sparse_structure(cond, batch_size, { "noise_level": 0.0 })

            slats = self.pipeline.sample_slat(cond, coords)
            meshes = [ self.pipeline.decode_slat(slat, ["mesh"])["mesh"][0] for slat in slats ]
            meshes = [ to_trimesh(mesh) for mesh in meshes ]
            meshes = [ post_process_mesh(mesh) for mesh in meshes ]

            objective_values = self.objective_evaluator(meshes)
            objective_values = objective_values.to(self.device)

            # each element is len of batch_size
            training_data.append({
                "x0": collected_data[-1]["return"]["pred_x_prev"],
                "objective_values": objective_values,
            })
            
            all_meshes.append(meshes)
            all_slats.append(slats)
            del collected_data
        
        training_data = {
            key: concat([batch[key] for batch in training_data])
            for key in training_data[0].keys()
        }

        objective_values = training_data["objective_values"]
        gathered_objective_values = self.accelerator.gather(objective_values)
        gathered_rewards = - gathered_objective_values
        gathered_advantages = ( gathered_rewards - gathered_rewards.mean() ) / (gathered_rewards.std(unbiased=False) + 1e-4) # unbiased=False to match np.std() in the official source code

        training_data["advantages"] = einops.rearrange(gathered_advantages,'(process batch) -> process batch',process=self.accelerator.num_processes)[self.accelerator.process_index]

        self.log_objective_metrics(objective_values=objective_values, objective_evaluations=self.config.training_samples_per_epoch * epoch, stage="train")

        return training_data

    def training_step(self, epoch, training_data):
        self.pipeline.models['sparse_structure_flow_model'].train()

        training_batches = list(batches_dict(training_data, self.config.training_batch_size_per_device))
        
        rescale_t = self.pipeline.sparse_structure_sampler_params["rescale_t"]
        timesteps = np.linspace(1, 0, self.config.num_inference_steps + 1)
        timesteps = rescale_t * timesteps / (1 + (rescale_t - 1) * timesteps)
        timesteps = torch.from_numpy(timesteps).to(self.device)

        for training_batch in tqdm(training_batches, desc="Training Batches", position=1, leave=False, disable=not self.accelerator.is_main_process):

            for i,t in tqdm(enumerate(timesteps), desc="Timesteps", position=2, leave=False, disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(self.pipeline.models['sparse_structure_flow_model']):
                    
                    x0 = training_batch["x0"]
                    x1 = torch.randn_like(x0)
                    xt = t*x1 + (1-t)*x0
                    cond = self.pipeline.get_cond([self.prompt]*len(xt))["cond"]

                    with torch.no_grad(), use_adapter(self.accelerator.unwrap_model(self.pipeline.models['sparse_structure_flow_model']), "old"):
                        old_flow_pred = self.pipeline.models['sparse_structure_flow_model'](xt, t.expand(xt.shape[0]), cond)

                    with torch.enable_grad():
                        curr_flow_pred = self.pipeline.models['sparse_structure_flow_model'](xt, t.expand(xt.shape[0]), cond)
                    
                    with self.accelerator.unwrap_model(self.pipeline.models['sparse_structure_flow_model']).disable_adapter(), torch.no_grad():
                        ref_flow_pred = self.pipeline.models['sparse_structure_flow_model'](xt, t.expand(xt.shape[0]), cond)

                    advantages = torch.clamp(training_batch["advantages"],-self.config.adv_clip_max,self.config.adv_clip_max)
                    normalized_advantages_clip = (advantages / self.config.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    positive_x0_pred = xt - t * curr_flow_pred
                    negative_x0_pred = xt - t * (2 * old_flow_pred - curr_flow_pred)
                    with torch.no_grad():
                        positive_weight_factor = (torch.abs(positive_x0_pred.to(torch.float64) - x0.to(torch.float64)).mean(dim=(1,2,3), keepdim=True).clip(min=0.00001))
                        negative_weight_factor = (torch.abs(negative_x0_pred.to(torch.float64) - x0.to(torch.float64)).mean(dim=(1,2,3), keepdim=True).clip(min=0.00001))
                    positive_loss = ((positive_x0_pred - x0) ** 2 / positive_weight_factor).mean(dim=(1,2,3))
                    negative_loss = ((negative_x0_pred - x0) ** 2 / negative_weight_factor).mean(dim=(1,2,3))
                    
                    policy_loss = (r * positive_loss + (1.0 - r) * negative_loss) * self.config.adv_clip_max

                    kl_loss = ((curr_flow_pred - ref_flow_pred) ** 2).mean(dim=(1,2,3))
                    loss = policy_loss.mean() + self.config.kl_beta * kl_loss.mean()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.pipeline.models['sparse_structure_flow_model'].parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
        ema_step = epoch
        ema_decay = min(0.001 * ema_step, 0.5)
        self.copy_parameters(sources=self.trainable_parameters, destinations=self.old_trainable_parameters, ema_decay=ema_decay)

    @torch.no_grad()
    def evaluation_step(self, epoch):
        
        self.pipeline.models['sparse_structure_flow_model'].eval()
        all_objective_values = []
        all_meshes = []
        all_slats = []
        for data_ids in tqdm(self.eval_dataloader, desc="Evaluation", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            batch_size = len(data_ids)
            eval_noise = torch.load("eval-noise/struct-tensor.pt", map_location=self.device)[:, data_ids, :]

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
            objective_values = objective_values.to(self.device)

            all_objective_values.append(objective_values)
            all_meshes.extend(meshes)
            all_slats.extend(slats)

        all_objective_values = torch.cat(all_objective_values, dim=0)

        objective_evaluations = self.config.training_samples_per_epoch * epoch
        self.log_meshes(all_meshes, all_slats, all_objective_values, objective_evaluations=objective_evaluations, stage="eval")
        self.log_objective_metrics(all_objective_values, objective_evaluations=objective_evaluations, stage="eval")
        self.accelerator.wait_for_everyone()



if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
