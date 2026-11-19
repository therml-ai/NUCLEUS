import itertools

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, ConstantLR, LinearLR


class CosineWarmupLR(SequentialLR):
    """
    Cosine learning rate scheduler with linear warmup
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        warmup_iters (int): Number of warmup iterations
        max_iters (int): Maximum number of iterations
        eta_min (float): Minimum learning rate
    """
    def __init__(self, optimizer, warmup_iters, max_iters, eta_min=0, last_epoch=-1):
        super().__init__(
            optimizer,
            schedulers=[
                # step + 1, so the first step trains rather than running at lr = 0, and
                # the max keeps a warmup of 0 iterations from dividing by zero
                LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: min(1.0, (step + 1) / max(warmup_iters, 1))
                ),
                # the cosine starts after the warmup, so it spans only the remaining
                # iterations and reaches eta_min at max_iters
                CosineAnnealingLR(
                    optimizer,
                    T_max=max_iters - warmup_iters,
                    eta_min=eta_min,
                    last_epoch=last_epoch
                ),
            ],
            milestones=[warmup_iters],
            last_epoch=last_epoch,
        )

class TrapezoidalLR(SequentialLR):
    """
    Trapezoidal learning rate scheduler
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        warmup_iters (int): Number of warmup iterations
        max_iters (int): Maximum number of iterations
        eta_min (float): Minimum learning rate
    """
    def __init__(self, optimizer, scale_factor, warmup_iters, flat_iters, cooldown_iters, eta_min=0, last_epoch=-1):
        phases = [
            (warmup_iters, lambda: LinearLR(optimizer, start_factor=scale_factor, total_iters=warmup_iters)),
            (flat_iters, lambda: ConstantLR(optimizer, factor=1.0, total_iters=flat_iters)),
            (cooldown_iters, lambda: LinearLR(
                optimizer, start_factor=1.0, end_factor=scale_factor, total_iters=cooldown_iters
            )),
        ]
        # a phase of no iterations is left out rather than built: LinearLR divides by
        # its total_iters, and each scheduler changes the learning rate when built
        lengths = [length for length, _ in phases if length > 0]
        super().__init__(
            optimizer,
            schedulers=[build() for length, build in phases if length > 0],
            milestones=list(itertools.accumulate(lengths[:-1])),
            last_epoch=last_epoch,
        )