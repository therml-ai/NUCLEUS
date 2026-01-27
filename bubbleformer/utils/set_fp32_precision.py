import torch

def set_fp32_precision():
    r"""
    This uses IEEE float32 precision for all operations.
    We want the input and output to always be in float32 and fully accumulated in float32.
    I.e., the encoder and decoder may use float32, but attention and MLP blocks explicitly use bfloat16.
    """
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"