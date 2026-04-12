import torch

def histogram(indices, num_buckets):
    # Wrapper for histc since, 1. histc seems better on cuda, but is not supported on CPUs.
    if indices.device.type == "cuda":
        return torch.histc(indices, min=0, max=num_buckets - 1, bins=num_buckets)
    else:
        return torch.bincount(indices, minlength=num_buckets)