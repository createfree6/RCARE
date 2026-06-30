from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        checkpoint_path = Path(path) / "checkpoint.pth"
        save_path = checkpoint_path.resolve()
        if os.name == "nt":
            save_path_str = str(save_path)
            if not save_path_str.startswith("\\\\?\\"):
                save_path = Path("\\\\?\\" + save_path_str)
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, args) -> None:
    if getattr(args, "lradj", "type1") == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif getattr(args, "lradj", "type1") == "constant":
        lr_adjust = {}
    else:
        lr_adjust = {}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")
