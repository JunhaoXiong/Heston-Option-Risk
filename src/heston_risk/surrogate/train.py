"""Training loop: MSE on implied vol, Adam, step-down learning rate on plateau, early stopping."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from heston_risk.surrogate.model import IVNet, Surrogate, scale_inputs


@dataclass
class TrainConfig:
    width: int = 128
    depth: int = 4
    batch_size: int = 1024
    lr: float = 1e-3
    max_epochs: int = 150
    patience: int = 10          # early stopping on validation RMSE
    lr_patience: int = 3        # halve the learning rate after this many epochs without improvement
    min_lr: float = 1e-5
    seed: int = 0
    threads: int | None = None


@dataclass
class TrainLog:
    epoch: list = field(default_factory=list)
    train_rmse: list = field(default_factory=list)
    val_rmse: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    seconds: float = 0.0
    best_epoch: int = 0


def train_surrogate(X_tr, y_tr, X_va, y_va, cfg: TrainConfig = TrainConfig(), verbose: bool = True):
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    xt = torch.from_numpy(scale_inputs(X_tr))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    xv = torch.from_numpy(scale_inputs(X_va))
    yv = torch.from_numpy(y_va.astype(np.float32))
    model = IVNet(cfg.width, cfg.depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg.lr_patience,
                                                       min_lr=cfg.min_lr)
    log = TrainLog()
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(cfg.seed)
    t0 = time.perf_counter()
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        tot, n = 0.0, 0
        for i in range(0, len(xt), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = torch.mean((model(xt[idx]) - yt[idx]) ** 2)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        model.eval()
        with torch.no_grad():
            val = torch.sqrt(torch.mean((model(xv) - yv) ** 2)).item()
        sched.step(val)
        log.epoch.append(ep)
        log.train_rmse.append(float(np.sqrt(tot / n)))
        log.val_rmse.append(val)
        log.lr.append(opt.param_groups[0]["lr"])
        if verbose:
            print(f"epoch {ep:3d}  train {np.sqrt(tot / n) * 100:.4f}  val {val * 100:.4f} vol pts  "
                  f"lr {opt.param_groups[0]['lr']:.1e}", flush=True)
        if val < best - 1e-7:
            best, bad, log.best_epoch = val, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    log.seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return Surrogate(model), log
