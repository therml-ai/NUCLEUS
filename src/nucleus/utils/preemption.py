import os
import random
import threading
import signal
from pathlib import Path
from types import FrameType

import numpy as np
import torch


class PreemptionHandler:
    def __init__(self) -> None:
        self.requested = False
        self.previous_handlers = {}
        self.shutdown_timer = None

    def __enter__(self):
        for signal_number in (signal.SIGTERM, signal.SIGUSR1):
            self.previous_handlers[signal_number] = signal.signal(
                signal_number, self.handle_preemption
            )
        return self

    def handle_preemption(self, signum: int, frame: FrameType | None) -> None:
        # Defer checkpoint IO until the optimizer update is complete.
        if self.requested:
            return
        self.requested = True
        print("Preemption received; checkpointing and exiting within 90 seconds.", flush=True)
        # A stuck GPU operation or checkpoint write must not delay shutdown indefinitely.
        self.shutdown_timer = threading.Timer(90, lambda: os._exit(1))
        self.shutdown_timer.daemon = True
        self.shutdown_timer.start()

    def __exit__(self, *exception_info) -> None:
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
        for signal_number, handler in self.previous_handlers.items():
            signal.signal(signal_number, handler)


def save_training_checkpoint(
    path: Path, model, optimizer, scheduler, global_step: int, epoch: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def restore_training_checkpoint(path: Path, model, optimizer, scheduler) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(f"Resumed diffusion at step {state['global_step']} from {path}", flush=True)
    return state["global_step"], state["epoch"]
