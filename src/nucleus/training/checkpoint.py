import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from nucleus.training.optim import Optimization

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"


@dataclass
class TrainingState:
    step: int = 0
    best_val_loss: float = math.inf


class CheckpointManager:
    """Writes the last checkpoint and the one with the lowest validation loss. A
    checkpoint holds the model's state dict, whose extra state rebuilds the model
    and its normalizer (see ``nucleus.models.load_model``), the optimizer and
    scheduler states, the training state, and the run's config."""

    def __init__(self, directory: Union[str, Path], model: nn.Module, optimization: Optimization, config: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimization = optimization
        self.config = config

    def save_last(self, state: TrainingState) -> Path:
        return self._save(LAST_CHECKPOINT, state)

    def save_if_best(self, val_loss: float, state: TrainingState) -> Optional[Path]:
        if val_loss >= state.best_val_loss:
            return None
        state.best_val_loss = val_loss
        return self._save(BEST_CHECKPOINT, state)

    def _save(self, name: str, state: TrainingState) -> Path:
        path = self.directory / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimization": self.optimization.state_dict(),
                "state": dataclasses.asdict(state),
                "config": self.config,
            },
            path,
        )
        return path


def restore_training(
    path: Union[str, Path], model: nn.Module, optimization: Optimization, map_location=None
) -> TrainingState:
    """Loads a checkpoint into a built model and its optimization, to resume a run."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimization.load_state_dict(checkpoint["optimization"])
    return TrainingState(**checkpoint["state"])
