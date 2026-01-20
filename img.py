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
from scipy.optimize import linear_sum_assignment
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
import pandas as pd
import plotly.express as px
from pymoo.indicators.hv import HV


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/img.py", "Training configuration.")

def get_frac(x):
    """
    Calculates the fractional part of a number.
    """
    return x - np.floor(x)

def is_prime(num):
    """
    Checks if a number is prime.
    """
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def find_next_prime(num):
    """
    Finds the smallest prime number greater than or equal to the given number.
    """
    prime_candidate = num
    while True:
        if is_prime(prime_candidate):
            return prime_candidate
        prime_candidate += 1

# From paper: [Relationships between Decomposition-based MOEAs and Indicator-based MOEAs], Algorithm 6
def generate_weight_vectors(N, n):
    """
    Implements Algorithm 6 from the paper to generate weight vectors.

    Args:
        N (int): The number of weight vectors to generate.
        n (int): The dimension of the objective space.

    Returns:
        numpy.ndarray: A set of weight vectors W. Note the potential for
                       dimension mismatch due to bugs in the source algorithm.
    """
    # --- Step 1: Find the smallest prime number p ---
    # Find the smallest prime number p that satisfies p >= 2*(n-1)+1
    p = find_next_prime(2 * (n - 1) + 1)
    print(f"Step 1: Using prime number p = {p}")

    # --- Step 2: Construct the generating vector z ---
    # The paper's notation {x} is interpreted as the fractional part (x mod 1),
    # and [x] is interpreted as rounding to the nearest integer.
    z = np.zeros(n - 1)
    z[0] = 1
    for i in range(1, n - 2 + 1): # Corresponds to indices 1 to n-2 in paper
        expr = 2 * np.cos(2 * np.pi * i / p)
        # The paper's notation is ambiguous. We interpret [N * {expr}] as
        # rounding (N times the fractional part of expr).
        fractional_part = get_frac(expr)
        z[i] = np.round(N * fractional_part)
    print(f"Step 2: Generating vector z = {z}")

    # --- Step 3-5: Generate N points T in the (n-1) dimensional unit cube ---
    T = np.zeros((N, n - 1))    

    for j in range(1, N + 1):
        # T_j = {(j * z) / N}, where {...} is the fractional part.
        # This can be calculated efficiently using the modulo operator.
        T[j-1, :] = get_frac(j * z / N)
    print(f"Step 3-5: Generated T matrix of shape {T.shape}")

    U = np.random.rand(1, (n - 1))
    T = np.remainder((T + U), 1)

    # --- Step 6-9: Project T into subspaces Theta and X ---
    q = math.ceil((n - 1) / 2)
    # Theta corresponds to the first q columns of T
    Theta = T[:, 0:q]
    # X corresponds to the remaining columns
    X = T[:, q:n-1]
    # Scale Theta
    Theta = (np.pi / 2) * Theta
    print(f"Step 6-9: Created Theta (shape {Theta.shape}) and X (shape {X.shape})")

    # --- Step 10: Define k ---
    k = math.floor(n / 2)
    print(f"Step 10: k = {k}")

    # --- Step 11-31: Construct Weight Vectors W ---

    if n % 2 == 0:
        # --- Even n case ---
        print("Executing odd n case...")
        # Initialize W. NOTE: The algorithm only seems to fill n-1 components.
        
        
        Y = np.zeros((k+1, N))
        Y[0,:] = 0
        Y[k,:] = 1

        W = np.zeros((n, N))

        for i in range(k-1, 0, -1):
            Y[i] = Y[i+1] * (X[:,i-1] ** (1 / i))

        # Loop for i = 1 to k
        for i in range(1, k + 1):
            # W_{2i-1} = sqrt(Y_i - Y_{i-1}) * cos(Theta_i)
            # W_{2i}   = sqrt(Y_i - Y_{i-1}) * sin(Theta_i)
            # Python indices: W[:, 2*i-2], W[:, 2*i-1], Theta[:, i-1]
            term = np.sqrt(Y[i] - Y[i - 1])
            W[2 * i - 2] = term * np.cos(Theta[:, i - 1])
            W[2 * i - 1] = term * np.sin(Theta[:, i - 1])

    else:
        # --- Odd n case ---

        Y = np.zeros((k+1, N))
        Y[k,:] = 1

        W = np.zeros((n, N))

        # Loop for i = k down to 1
        for i in range(k, 0, -1):
            # Y_i = Y_{i+1} * X_{i,:}^{2/(2i-1)}
            # Paper uses 1-based indexing for X. X_{i,:} -> X[:, i-1]
            Y[i-1] = Y[i] * (X[:, i - 1] ** (2 / (2 * i - 1)))

        W[0] = np.sqrt(Y[0])

        # Loop for i = 1 to k
        for i in range(1, k + 1):
            # W_{2i}   = sqrt(Y_{i+1} - Y_i) * cos(Theta_i)
            # W_{2i+1} = sqrt(Y_{i+1} - Y_i) * sin(Theta_i)
            # Python indices: W[:, 2*i-1], W[:, 2*i], Theta[:, i-1]
            term = np.sqrt(Y[i] - Y[i-1])
            W[2 * i - 1] = term * np.cos(Theta[:, i - 1])
            W[2 * i]     = term * np.sin(Theta[:, i - 1])

    # --- Step 32: Return W ---
    return W

class Trainer(BaseTrainer):
    def __init__(self, config):

        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, "Only supports single GPU"

        super().__init__(config)

        self.pipeline.sparse_structure_sampler.sample_once_original = self.pipeline.sparse_structure_sampler.sample_once
        self.pipeline.sparse_structure_sampler.sample_once = self.sample_once_img.__get__(self.pipeline.sparse_structure_sampler)

        self.num_objectives = 2

        # preference vector
        weight_vectors = generate_weight_vectors(self.config.batch_size, self.num_objectives)
        self.weight_vectors = torch.from_numpy(weight_vectors.T).to(self.accelerator.device)


    @torch.inference_mode()
    def run(self) -> None:

        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": 0.7,
            "external_self": self,
        }

        cond = self.pipeline.get_cond([self.prompt]*self.config.batch_size)
        coords = self.pipeline.sample_sparse_structure(cond, self.config.batch_size, sparse_structure_sampler_params)
        meshes, slats = self.generate_meshes_from_coords(cond, coords)

        objective_values = self.objective_evaluator(meshes)
        objective_values = objective_values.to(self.device)
        total_objective_evaluations = self.config.batch_size * self.config.expansion_size * self.config.num_inference_steps
        self.log_objective_metrics(objective_values, objective_evaluations=total_objective_evaluations, stage="final")
        self.log_meshes(meshes, slats, objective_values, objective_evaluations=total_objective_evaluations, stage="final")

    @staticmethod
    def sample_once_img(
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
        To replace the original sample_once with img sampling
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
        
        # ------------ img code ------------- #
        x_prev_candidates = []
        x_prev_candidates_multi_obj_values = []
        
        for i, x_prev_mean_i in enumerate(x_prev_mean):
            # sample
            noise_i = torch.randn( (external_self.config.expansion_size,) + x_prev_mean_i.shape, device=x_prev_mean_i.device)
            x_prev_i = x_prev_mean_i + std_dev_t * np.sqrt(-1*dt) * noise_i
            x_prev_candidates.append(x_prev_i)
            
            # determinisitcally sample once, decode and evaluate
            multi_objective_values = torch.full((external_self.config.expansion_size,external_self.num_objectives), float('inf'), device=x_t.device)
            for j, x_prev_ij in enumerate(x_prev_i):
                try:
                    pred_sample_ij = self.sample_once_original(model, x_prev_ij.unsqueeze(0), t_prev, t_seq[t_prev_prev_idx], cond_one, noise_level=0.0, **kwargs).pred_x_0 if t_prev_prev_idx < len(t_seq) else x_prev_ij.unsqueeze(0)
                    coords = torch.argwhere(external_self.pipeline.models['sparse_structure_decoder'](pred_sample_ij)>0)[:, [0, 2, 3, 4]].int()
                    meshes, slats = external_self.generate_meshes_from_coords(cond_dict_one, coords)
                    multi_objective_values[j] = external_self.objective_evaluator(meshes).to(x_t.device)
                except Exception as e:
                    print(f"Exception {e} when decoding at {t_idx}-th timestep, assign inf objective values")
            x_prev_candidates_multi_obj_values.append(multi_objective_values)

        # flatten
        x_prev_candidates = torch.cat(x_prev_candidates, dim=0)
        x_prev_candidates_multi_obj_values = torch.cat(x_prev_candidates_multi_obj_values, dim=0) # (batch size * expansion size, num objectives)

        if x_prev_candidates_multi_obj_values.isinf().any().item():
            selected_ids = torch.arange(0, x_prev_candidates_multi_obj_values.shape[0], step=external_self.config.expansion_size, device=x_t.device).tolist()
        else:
            selected_ids = []
            available_indices = list(range(x_prev_candidates_multi_obj_values.shape[0]))
            for weight_idx in torch.randperm(external_self.weight_vectors.shape[0]):
                local_idx = external_self.select_candidate(x_prev_candidates_multi_obj_values[available_indices], external_self.weight_vectors[weight_idx])
                global_idx = available_indices.pop(local_idx)
                selected_ids.append(global_idx)
            selected_ids = torch.tensor(selected_ids, device=x_t.device)

        x_prev = x_prev_candidates[selected_ids]
        selected_multi_obj_values = x_prev_candidates_multi_obj_values[selected_ids]

        # log
        if torch.isfinite(selected_multi_obj_values).all():
            external_self.log_objective_metrics(selected_multi_obj_values, objective_evaluations=external_self.config.batch_size * external_self.config.expansion_size * (t_idx+1))
            external_self.log_selection(x_prev_candidates_multi_obj_values, torch.tensor(selected_ids, device=x_t.device), t_idx)

        return edict({"pred_x_prev": x_prev, "pred_x_0": pred_x_0})

    def select_candidate(objective_values, preference_vector):
        """
        Selects the best candidate based on the preference vector.
        
        Args:
            objective_values: (N, K) torch.Tensor.
            preference_vector: (K,) torch.Tensor.
            
        Returns:
            int: The index of the selected candidate.
        """

        # normalize to [-1,0]
        objective_values = (objective_values - objective_values.min(dim=0).values) / (objective_values.max(dim=0).values - objective_values.min(dim=0).values) - 1.0
        aggregated_values = torch.max(objective_values / preference_vector[None, :], dim=1).values
        best_idx = torch.argmin(aggregated_values).item()

        return best_idx

    def log_selection(self, candidates_multi_obj_values, selected_ids, t_idx):
        vals = candidates_multi_obj_values.detach().cpu().numpy()
        labels = np.full(vals.shape[0], "Candidate")
        labels[selected_ids.detach().cpu().numpy()] = "Selected"

        df = pd.DataFrame({self.objective_evaluator.objective_short_names[0]: vals[:, 0], self.objective_evaluator.objective_short_names[1]: vals[:, 1], "label": labels})
        fig = px.scatter(df, x=self.objective_evaluator.objective_short_names[0], y=self.objective_evaluator.objective_short_names[1], color="label")
        wandb.log({"selection-scatter": fig, "timestep": t_idx })

    def select_candidate(self, objective_values, preference_vector):
        """
        Selects the best candidate based on the preference vector.
        
        Args:
            objective_values: (N, K) torch.Tensor.
            preference_vector: (K,) torch.Tensor.
            
        Returns:
            int: The index of the selected candidate.
        """

        # normalize to [-1,0]
        objective_values = (objective_values - objective_values.min(dim=0).values) / (objective_values.max(dim=0).values - objective_values.min(dim=0).values) - 1.0
        aggregated_values = torch.max(objective_values / preference_vector[None, :], dim=1).values
        best_idx = torch.argmin(aggregated_values).item()

        return best_idx


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()