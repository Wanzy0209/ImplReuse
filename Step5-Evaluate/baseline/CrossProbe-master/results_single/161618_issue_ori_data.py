# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import triton
import torch

import torch._inductor.config as inductor_config

m = 20120
k = 1536
n = 512

a = torch.randn((m, n)).requires_grad_(False).cuda()
mat1 = torch.randn((m, k)).requires_grad_(False).cuda()
mat2 = torch.randn((k, n)).requires_grad_(False).cuda()
f = lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)

with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,):
    compiled = torch.compile(f, dynamic=False)
    compiled(a, mat1, mat2)