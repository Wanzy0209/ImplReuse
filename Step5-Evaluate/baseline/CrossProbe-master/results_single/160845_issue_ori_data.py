# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

torch.manual_seed(0)
device = "mps"

shape = (400, )
idx = torch.arange(400, dtype=torch.long)
dim = 0


t_mps = torch.zeros(shape, dtype=torch.complex64, device=device)
t_cpu = torch.zeros(shape, dtype=torch.complex64, device="cpu")
trailing = shape[dim+1:]
src_shape = (len(idx),) + trailing
src_imag = torch.randn(src_shape, dtype=torch.float32, device=device)
src_real = torch.zeros_like(src_imag)
src = torch.complex(src_real, src_imag)
t_mps.index_add_(dim, idx.to(device), src)
t_cpu.index_add_(dim, idx.cpu(), src.cpu())

print("MPS imag sum:", t_mps.imag.abs().sum().item())
print("CPU imag sum:", t_cpu.imag.abs().sum().item())
print("max abs diff:", (t_mps.cpu() - t_cpu).abs().max().item())