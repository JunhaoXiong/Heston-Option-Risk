r"""MLP surrogate: (v0, kappa, theta, sigma, rho, k, T) -> Black-Scholes implied vol.

Why an MLP fits this problem
* The map from (parameters, strike, maturity) to IV is a fixed, smooth, *deterministic*
  function: the analytic pricer computes it exactly, just slowly (an integral per maturity).
  There is no noise and no time series; the network only has to approximate a known function.
* A multilayer perceptron with smooth activations is a universal approximator of smooth
  functions, and its evaluation is a handful of small matrix products, independent of the
  maturity grid or Fourier integration.
* The inputs interact non-linearly (e.g. rho mostly moves the skew, sigma the curvature, both
  more at short T); stacked non-linear layers learn such interactions without hand-crafted features.

Input scaling (fixed, from the sampling box, never fitted on data, so nothing leaks from test):
  each parameter -> [-1, 1] over its box;  z = k / (0.6 sqrt(T)) in [-1, 1];  tau = log T -> [-1, 1].
Using z instead of k makes every maturity use the same input range.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from heston_risk.surrogate.dataset import BOX_HI, BOX_LO, K_WIDTH, T_MAX, T_MIN


def scale_inputs(X: np.ndarray) -> np.ndarray:
    """X columns: v0, kappa, theta, sigma, rho, k, T  ->  7 features in about [-1, 1]."""
    X = np.asarray(X, dtype=np.float64)
    p = 2.0 * (X[:, :5] - BOX_LO) / (BOX_HI - BOX_LO) - 1.0
    T = X[:, 6]
    z = X[:, 5] / (K_WIDTH * np.sqrt(T))
    tau = 2.0 * (np.log(T) - np.log(T_MIN)) / (np.log(T_MAX) - np.log(T_MIN)) - 1.0
    return np.column_stack([p, z, tau]).astype(np.float32)


def in_domain(X: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Rows inside the training box (the network is not trusted outside it)."""
    S = scale_inputs(X)
    return np.all(np.abs(S) <= 1.0 + tol, axis=1)


class IVNet(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4):
        super().__init__()
        layers, d = [], 7
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Surrogate:
    """Numpy-in / numpy-out wrapper around a trained IVNet."""

    def __init__(self, model: IVNet):
        self.model = model.eval()

    @torch.no_grad()
    def iv(self, X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(scale_inputs(X))
        return self.model(x).double().numpy()

    def iv_and_param_jacobian(self, X: np.ndarray):
        """IV and its exact derivative with respect to the 5 Heston parameters (autograd).

        Finite differences on a float32 network are unreliable (output noise ~1e-7 swamps a tiny
        bump), so calibration with the surrogate uses this analytic network gradient instead.
        """
        x = torch.from_numpy(scale_inputs(X)).requires_grad_(True)
        out = self.model(x)
        (g,) = torch.autograd.grad(out.sum(), x)  # rows are independent -> per-row gradient
        dscale = 2.0 / (BOX_HI - BOX_LO)            # d(scaled param)/d(param)
        jac = g[:, :5].double().numpy() * dscale
        return out.detach().double().numpy(), jac

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(),
                    "width": self.model.net[0].out_features,
                    "depth": sum(isinstance(m, nn.Linear) for m in self.model.net) - 1}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m = IVNet(ck["width"], ck["depth"])
        m.load_state_dict(ck["state_dict"])
        return cls(m)
