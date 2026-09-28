import pytest
import torch

from nucleus.utils.metrics import precision_recall
from nucleus.utils.physical_metrics import nucleation_event_masks


def test_precision_recall_matches_manual():
    pred = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    # tp=1, fp=1, fn=1
    precision, recall = precision_recall(pred, target)
    assert precision.item() == pytest.approx(0.5)
    assert recall.item() == pytest.approx(0.5)


def test_precision_recall_empty_is_zero_not_nan():
    zeros = torch.zeros(4, dtype=torch.int32)
    precision, recall = precision_recall(zeros, zeros)
    assert torch.isfinite(precision) and precision.item() == pytest.approx(0.0)
    assert torch.isfinite(recall) and recall.item() == pytest.approx(0.0)


def test_perfect_prediction_gives_unit_precision_recall():
    pred = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
    precision, recall = precision_recall(pred, pred)
    assert precision.item() == pytest.approx(1.0)
    assert recall.item() == pytest.approx(1.0)


def test_nucleation_masks_require_liquid_before():
    # prev: liquid(0)/vapor(1); only cells liquid-before can count as nucleation.
    prev = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    target = torch.tensor([1, 0, 1, 1], dtype=torch.int32)  # cell 0 nucleates
    pred = torch.tensor([1, 1, 1, 0], dtype=torch.int32)  # cell 0 correct, cell 1 false nucleation
    gt_nucleation, pred_nucleation = nucleation_event_masks(prev, target, pred)

    assert gt_nucleation.tolist() == [True, False, False, False]
    # cells 2/3 were vapor before, so a vapor prediction there is not nucleation
    assert pred_nucleation.tolist() == [True, True, False, False]

    # tp=1, fp=1, fn=0 -> precision 0.5, recall 1.0
    precision, recall = precision_recall(pred_nucleation, gt_nucleation)
    assert precision.item() == pytest.approx(0.5)
    assert recall.item() == pytest.approx(1.0)
