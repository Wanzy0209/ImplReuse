# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

torch.set_default_dtype(torch.bfloat16)
device = "cpu"
# device="cuda"
seqlens = torch.tensor([434,  78, 101, 411], device=device, dtype=torch.int32)
cu_seqlens = torch.cat((torch.zeros(1, device=device), torch.cumsum(seqlens, dim=0))).to(dtype=torch.int32, device=device)

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"{cu_seqlens=}")