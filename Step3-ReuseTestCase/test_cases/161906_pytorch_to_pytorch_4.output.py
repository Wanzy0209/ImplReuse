import torch
import torch.nn as nn
import torch.nn.functional as F

class M(nn.Module):
    def __init__(self, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
        # Introduce parameter for broadcasting / stride ops, similar to the original bug
        self.p = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (Batch, Channels, Height * r, Width * r)
        
        # Simulate operations that might affect strides/contiguity, 
        # similar to how S.real and S.imag are views in the original bug.
        # We transpose to make x non-contiguous, perform math, and transpose back.
        x = x.transpose(1, 2)  # (B, H*r, C, W*r)
        x = x / self.p         # scalar divide with Parameter
        x = x.transpose(1, 2)  # (B, C, H*r, W*r) - likely non-contiguous
        
        # Call the similar API: pixel_unshuffle
        # This replaces torch.complex(R, I) from the original bug
        y = F.pixel_unshuffle(x, self.downscale_factor)
        
        return y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Input suitable for pixel_unshuffle: (Batch, Channels, Height, Width)
    # downscale_factor=2, so Height and Width must be divisible by 2.
    # Shape (1, 3, 8, 8) -> Output (1, 12, 4, 4)
    x = torch.randn(1, 3, 8, 8, device=device)
    m = M(downscale_factor=2).to(device)

    # Eager: works fine
    y_eager = m(x)
    print("eager mode OK:", y_eager.shape)

    # Compile: check for stride assertion or runtime errors
    m_c = torch.compile(m)
    y_compiled = m_c(x)
    
    # Verify correctness
    assert torch.allclose(y_eager, y_compiled)
    print("compiled mode OK:", y_compiled.shape)

if __name__ == "__main__":
    main()