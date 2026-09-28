import torch

def sm120():
    """
    python -m natten.profiler -i 8 16 16 -d 512 -w 8 3 3 --heads 8 --dtype bf16 --backprop --optimize --optimize-warmup-steps 20
    This uses the kernel for sm80. NAtten seems to not have a specific kernel for sm120.
    """
    return dict(
        backend="cutlass-fna",
        q_tile_shape=(8, 2, 2),
        kv_tile_shape=(8, 4, 4),
        backward_q_tile_shape=(8, 2, 4),
        backward_kv_tile_shape=(8, 4, 2),
    )
    
def get_natten_config(device):
    try:
        torch.cuda.get_device_capability(device)
    except:
        return {}
    
    if device == "cpu" or not torch.cuda.is_available():
        return {}
    
    cuda_version = torch.cuda.get_device_capability(device)
    match cuda_version:
        case (12, _): return sm120()
        case _: return {}