import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import gpytorch
import torch.nn as nn
import einops
import time

class StdNormalizer:
    def __init__(self):
        self.mean = 0
        self.std = 1
    def fit_transform(self, x):
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (x - self.mean) / self.std
    def transform(self, x):
        return (x - self.mean) / self.std
    def inverse_transform(self, x):
        return x * self.std + self.mean

class ExactGpModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim):
        super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = (x_dim) ** 0.5

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ValueModel(nn.Module):
    def __init__(self, dimension, noise_level = 1e-2) -> None:
        super().__init__()
        self.dim = dimension
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_level
        self.likelihood.eval()

        model = ExactGpModel(None, None, self.likelihood, x_dim=dimension)

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self.y_normalizer = StdNormalizer()

        self.register_buffer("all_x", torch.empty(0, dimension, dtype=torch.float32))
        self.register_buffer("all_y", torch.empty(0, dtype=torch.float32))

    @torch.no_grad()
    def predict(self, x):
        device = x.device
        self.model.to(device)
        self.likelihood.to(device)
        
        with gpytorch.settings.fast_pred_var():
            y_preds = self.likelihood(self.model(x))
        
        y_preds_mean = self.y_normalizer.inverse_transform(y_preds.mean.to(device))
        y_preds_var = y_preds.variance.to(device)

        torch.cuda.empty_cache()

        return y_preds_mean.to(device), y_preds_var.to(device)

    # x: data points
    # y: lower is better
    def add_model_data(self, x, y):
        device = x.device

        MAX_SIZE = 1000

        self.all_x = torch.cat([self.all_x, x.to(torch.float32)], dim=0)[-MAX_SIZE:]
        self.all_y = torch.cat([self.all_y, y.to(torch.float32)], dim=0)[-MAX_SIZE:]
        
        self.model.set_train_data(
            inputs=self.all_x,
            targets=self.y_normalizer.fit_transform(self.all_y.unsqueeze(-1)).squeeze(-1),
            strict=False
        )

        torch.cuda.empty_cache()

