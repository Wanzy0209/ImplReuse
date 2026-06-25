import torch
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

def test_autocast_checkpointing():
    with autocast():
        x = torch.randn(2, 2, device='cuda')
        y = checkpoint(lambda t: t * 2, x)
    return y