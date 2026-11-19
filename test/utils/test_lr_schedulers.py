import pytest
import torch

from nucleus.utils.lr_schedulers import CosineWarmupLR

PEAK_LR, ETA_MIN = 3e-4, 1e-6


def _learning_rates(warmup_iters: int, max_iters: int) -> list:
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=PEAK_LR)
    scheduler = CosineWarmupLR(optimizer, warmup_iters=warmup_iters, max_iters=max_iters, eta_min=ETA_MIN)
    learning_rates = []
    for _ in range(max_iters):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return learning_rates


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cosine_warmup_peaks_after_the_warmup_and_decays_to_eta_min():
    learning_rates = _learning_rates(warmup_iters=50, max_iters=200)
    assert 0 < learning_rates[0] < PEAK_LR
    assert learning_rates[49] == pytest.approx(PEAK_LR)
    assert max(learning_rates) == pytest.approx(PEAK_LR)
    assert learning_rates[-1] == pytest.approx(ETA_MIN, rel=0.05)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cosine_warmup_without_warmup_starts_at_the_peak():
    learning_rates = _learning_rates(warmup_iters=0, max_iters=100)
    assert learning_rates[0] == pytest.approx(PEAK_LR)
    assert learning_rates[-1] == pytest.approx(ETA_MIN, rel=0.5)
