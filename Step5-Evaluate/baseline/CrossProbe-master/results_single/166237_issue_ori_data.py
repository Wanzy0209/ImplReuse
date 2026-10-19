# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

x_cpu = torch.randn((5, ), requires_grad=True)
src_cpu = torch.randn((5,), requires_grad=True)
index = torch.randint(0, 5, (5,))

x_mps = x_cpu.detach().to("mps").requires_grad_(True)
src_mps = src_cpu.detach().to("mps").requires_grad_(True)
index_mps = index.to('mps')

out_cpu = torch.index_copy(x_cpu, 0, index, src_cpu)
out_mps = torch.index_copy(x_mps, 0, index_mps, src_mps)

torch.testing.assert_close(out_cpu, out_mps.cpu())