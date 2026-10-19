# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

def f(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    z = torch.complex(real, imag)
    return torch.fft.irfft(z, dim=1)


B, F, T = 1, 641, 39

r_src = torch.randn(B, F, T)
i_src = torch.randn(B, F, T, )
r_mismatch = r_src.permute(0, 2, 1)
i_mismatch = i_src.permute(0, 2, 1)

compiled = torch.compile(f, fullgraph=True)

_ = compiled(r_src, i_src)
_ = compiled(r_mismatch, i_mismatch)