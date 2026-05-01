import sys
import numpy as np
import torch
from typing import Optional, Any
from easydict import EasyDict as edict
from absl import flags
from ml_collections import config_flags
from trainer.base_trainer import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/fk-steering.py", "Configuration.")


class Trainer(BaseTrainer):
    def __init__(self, config):
        assert torch.cuda.device_count() == 1, "Only supports single GPU"
        super().__init__(config)

        self.pipeline.sparse_structure_sampler.sample_once_original = self.pipeline.sparse_structure_sampler.sample_once
        self.pipeline.sparse_structure_sampler.sample_once = self.sample_once_fk_steering.__get__(self.pipeline.sparse_structure_sampler)

    @torch.inference_mode()
    def run(self) -> None:
        N = self.config.batch_size

        self.log_trajectory_obj_values = torch.zeros(
            (self.config.num_inference_steps, self.objective_evaluator.num_objectives),
            device=self.device,
        )
        self.max_r = torch.full((N,), float("-inf"), device=self.device, dtype=torch.float32)
        self._cond_dict_one = self.pipeline.get_cond([self.prompt])

        sparse_structure_sampler_params = {
            "steps": self.config.num_inference_steps,
            "noise_level": self.config.noise_level,
            "external_self": self,
        }

        cond = self.pipeline.get_cond([self.prompt] * N)
        coords = self.pipeline.sample_sparse_structure(cond, N, sparse_structure_sampler_params)
        meshes, slats = self.generate_meshes_from_coords(cond, coords)

        for t in range(self.config.num_inference_steps):
            objective_values_t = self.log_trajectory_obj_values[t:t+1, :]
            if torch.isfinite(objective_values_t).all():
                self.log_objective_metrics(objective_values_t, objective_evaluations=N * (t + 1))

        total_objective_evaluations = N * self.config.num_inference_steps
        final_obj_values = self.log_trajectory_obj_values[-1:, :]
        self.log_objective_metrics(final_obj_values, objective_evaluations=total_objective_evaluations, stage="final")
        self.log_meshes(meshes, slats, final_obj_values.expand(len(meshes), -1), objective_evaluations=total_objective_evaluations, stage="final")

    @staticmethod
    def sample_once_fk_steering(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        noise_level: float = 0.0,
        noise: Optional[torch.Tensor] = None,  # ignored
        external_self=None,
        **kwargs,
    ):
        """
        Replace the original sample_once with FK steering: propose with the standard SDE
        step, evaluate the reward at the current pred_x_0, track the running max reward
        per particle, and resample particles when ESS drops below half the population.
        """
        batch_size = x_t.shape[0]
        rescale_t = external_self.pipeline.sparse_structure_sampler_params["rescale_t"]
        t_seq = np.linspace(1, 0, external_self.config.num_inference_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_idx = np.argwhere(t_seq == t).item()
        cond_one = cond[0].unsqueeze(0)
        cond_dict_one = external_self._cond_dict_one
        assert torch.allclose(cond, cond_one.expand_as(cond))

        # ----------- standard SDE step ----------- #
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        dt = t_prev - t

        std_dev_t = np.sqrt(t / (1 - np.where(t == 1, t_prev, t))) * noise_level
        x_prev_mean = x_t * (1 + std_dev_t**2 / (2 * t) * dt) + pred_v * (1 + std_dev_t**2 * (1 - t) / (2 * t)) * dt

        # Propose: one fresh noise per particle
        et = torch.randn_like(x_t)
        x_prev = x_prev_mean + std_dev_t * np.sqrt(-1 * dt) * et

        # ------------ evaluate reward at pred_x_0 ------------- #
        objective_values = torch.full(
            (batch_size, external_self.objective_evaluator.num_objectives),
            float("inf"), device=x_t.device,
        )
        for j in range(batch_size):
            try:
                coords = torch.argwhere(external_self.pipeline.models['sparse_structure_decoder'](pred_x_0[j].unsqueeze(0)) > 0)[:, [0, 2, 3, 4]].int()
                meshes, _ = external_self.generate_meshes_from_coords(cond_dict_one, coords)
                objective_values[j] = external_self.objective_evaluator(meshes).to(x_t.device)
            except Exception as e:
                print(f"Exception {e} when decoding at {t_idx}-th timestep, particle {j}")

        # objective: lower is better. FK reward = -mean(objective).
        r = -objective_values.mean(dim=-1)
        external_self.max_r = torch.maximum(external_self.max_r, r)

        if torch.isfinite(objective_values).all():
            external_self.log_trajectory_obj_values[t_idx] = objective_values.mean(dim=0)

        # ------------ FK resampling ------------- #
        beta = external_self.config.beta
        finite_mask = torch.isfinite(external_self.max_r)
        if finite_mask.any():
            log_w = beta * external_self.max_r
            log_w = log_w - log_w[finite_mask].max()
            w = torch.where(finite_mask, torch.exp(log_w), torch.zeros_like(log_w))
            normalized_w = w / w.sum()
            ess = 1.0 / normalized_w.pow(2).sum()
            if ess < 0.5 * batch_size:
                idx = torch.multinomial(w, num_samples=batch_size, replacement=True)
                x_prev = x_prev[idx]
                pred_x_0 = pred_x_0[idx]
                external_self.max_r = external_self.max_r[idx]

        return edict({"pred_x_prev": x_prev, "pred_x_0": pred_x_0})


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
