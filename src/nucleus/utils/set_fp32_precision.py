import torch

def set_fp32_precision():
    # lightning does not like the new pytorch API, so we use the deprecated API
    # "high" enables using tf32.
    torch.set_float32_matmul_precision("high")
    #torch.backends.fp32_precision = "tf32"
    #torch.backends.cuda.matmul.fp32_precision = "tf32"
    #torch.backends.cudnn.fp32_precision = "tf32"
    #torch.backends.cudnn.conv.fp32_precision = "tf32"