import sys
import numpy as np
import torch
from typing import Optional, Any
from easydict import EasyDict as edict
from absl import flags
from ml_collections import config_flags
from trainer.base_trainer import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/evolvable.py", "Configuration.")


class Trainer(BaseTrainer):
    def __init__(self, config):
        assert torch.cuda.device_count() == 1, "Only supports single GPU"
        super().__init__(config)

        self.pipeline.sparse_structure_sampler.sample_once_original = self.pipeline.sparse_structure_sampler.sample_once
        self.pipeline.sparse_structure_sampler.sample_once = self.sample_once_evolvable.__get__(self.pipeline.sparse_structure_sampler)

    @torch.inference_mode()
    def run(self):
        self.log_trajectory_obj_values = torch.full(
            (self.config.num_inference_steps, self.objective_evaluator.num_objectives),
            float('inf'),
            device=self.device
        )
        self._cond_dict = self.pipeline.get_cond([self.prompt])

        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": self.config.noise_level,
            "external_self": self,
        }

        coords = self.pipeline.sample_sparse_structure(self._cond_dict, 1, sparse_structure_sampler_params)
        meshes, slats = self.generate_meshes_from_coords(self._cond_dict, coords)

        total_evals = self.config.perturbation_samples * self.config.num_inference_steps
        final_obj_values = self.log_trajectory_obj_values[-1:, :]
        self.log_objective_metrics(final_obj_values, objective_evaluations=total_evals, stage="final")
        self.log_meshes(meshes, slats, final_obj_values, objective_evaluations=total_evals, stage="final")

    @staticmethod
    def sample_once_evolvable(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        noise_level: float = 0.0,
        noise: Optional[torch.Tensor] = None,
        external_self=None,
        **kwargs
    ):
        n = external_self.config.perturbation_samples
        rescale_t = external_self.pipeline.sparse_structure_sampler_params["rescale_t"]
        t_seq = np.linspace(1, 0, external_self.config.num_inference_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_idx = np.argwhere(t_seq == t).item()

        cond_dict_one = external_self._cond_dict

        # Model prediction at current noisy state
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        dt = t_prev - t

        # SDE step coefficients
        std_dev_t = np.sqrt(t / (1 - np.where(t == 1, t_prev, t))) * noise_level
        x_prev_mean = x_t * (1 + std_dev_t**2 / (2 * t) * dt) + pred_v * (1 + std_dev_t**2 * (1 - t) / (2 * t)) * dt
        ct = std_dev_t * np.sqrt(-dt)  # noise scale for this step

        # Sample N perturbations and build candidates: x_prev_mean + ct * et
        # x_prev_mean is (1, D...), et is (n, D...), broadcasts to (n, D...)
        et = torch.randn((n,) + x_prev_mean.shape[1:], device=x_t.device, dtype=x_t.dtype)
        x_prev_candidates = x_prev_mean + ct * et  # (n, D...)

        # Evaluate each candidate via one-step lookahead to get pred_x_0, then decode
        objective_values = torch.full(
            (n, external_self.objective_evaluator.num_objectives), float('inf'), device=x_t.device
        )
        for j, x_prev_j in enumerate(x_prev_candidates):
            try:
                coords = torch.argwhere(external_self.pipeline.models['sparse_structure_decoder'](x_prev_j.unsqueeze(0)) > 0)[:, [0, 2, 3, 4]].int()
                meshes, _ = external_self.generate_meshes_from_coords(cond_dict_one, coords)
                objective_values[j] = external_self.objective_evaluator(meshes).to(x_t.device)
            except Exception as e:
                print(f"Exception at step {t_idx}, candidate {j}: {e}")

        rewards = -objective_values.mean(dim=-1)
        valid_mask = torch.isfinite(rewards)

        if not valid_mask.any():
            x_prev = x_prev_mean + ct * et[:1]
        else:
            external_self.log_trajectory_obj_values[t_idx] = objective_values[valid_mask].mean(dim=0)
            valid_rewards = rewards[valid_mask]
            valid_et = et[valid_mask]
            n_valid = valid_mask.sum().item()
            r = (valid_rewards - valid_rewards.mean()) / valid_rewards.std().clamp(min=1e-6)
            zt = torch.einsum("N, N ... -> ...", r.to(valid_et.dtype), valid_et) / n_valid ** 0.5
            x_prev = x_prev_mean + ct * zt.unsqueeze(0)

        if torch.isfinite(external_self.log_trajectory_obj_values[t_idx]).all():
            external_self.log_objective_metrics(external_self.log_trajectory_obj_values[t_idx:t_idx+1, :],objective_evaluations=(t_idx + 1) * n,)

        return edict({"pred_x_prev": x_prev, "pred_x_0": pred_x_0})


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
