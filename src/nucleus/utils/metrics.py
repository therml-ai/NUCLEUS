import torch

def precision_recall(pred_positive, target_positive, eps=1e-8):
    r"""
    Computes precision and recall between two boolean masks of identical shape.
    The eps guards the degenerate case where a class is never present and never
    predicted, in which case the corresponding metric is 0.
    """
    pred_positive = pred_positive.bool()
    target_positive = target_positive.bool()
    true_positive = (pred_positive & target_positive).sum().to(torch.float32)
    false_positive = (pred_positive & ~target_positive).sum().to(torch.float32)
    false_negative = (~pred_positive & target_positive).sum().to(torch.float32)
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    return precision, recall
