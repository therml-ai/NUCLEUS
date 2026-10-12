from typing import Mapping, Optional, Union

import torch

Metric = Union[float, torch.Tensor]


class MetricLogger:
    """Logs scalar metrics to wandb when a run is given, and the headline ones to
    stdout."""

    def __init__(self, run=None):
        self.run = run

    @classmethod
    def wandb(cls, project: str, entity: str, name: str, directory: str, config: dict) -> "MetricLogger":
        import wandb

        return cls(wandb.init(project=project, entity=entity, name=name, dir=directory, config=config))

    def log(self, metrics: Mapping[str, Metric], step: int) -> None:
        values = {name: _to_float(value) for name, value in metrics.items()}
        if self.run is not None:
            self.run.log(values, step=step)
        # the per-layer MoE statistics would bury the losses
        headline = " ".join(
            f"{name}={value:.4g}" for name, value in sorted(values.items()) if "_moe/" not in name
        )
        print(f"step {step}: {headline}", flush=True)

    def finish(self, failed: bool = False) -> None:
        if self.run is not None:
            # a nonzero exit code marks the wandb run as failed
            self.run.finish(exit_code=1 if failed else 0)


def _to_float(value: Metric) -> float:
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
