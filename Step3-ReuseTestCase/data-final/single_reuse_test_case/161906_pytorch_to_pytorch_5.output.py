import torch
import torch.nn as nn
import torch.nn.functional as F

class M(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        # Introduce parameter for broadcasting / stride ops, similar to the original bug context
        self.p = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, C * r^2, H, W)
        # Apply operation involving parameter to potentially affect strides/contiguity
        x = x / self.p
        
        # Call the similar API: torch.nn.functional.pixel_shuffle
        # This operation involves view, permute, and reshape, which are sensitive to strides
        return F.pixel_shuffle(x, self.upscale_factor)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Input shape: (Batch=1, Channels=4 (r^2), H=8, W=8)
    # r=2, so C_in = 4, C_out = 1
    x = torch.randn(1, 4, 8, 8, device=device)
    m = M().to(device)

    # Eager: works fine
    z_eager = m(x)
    print("eager mode OK:", z_eager.shape)

    # Compile: check for runtime stride assertion or other errors
    m_c = torch.compile(m)
    z_compiled = m_c(x)
    
    # Verify results match
    assert torch.allclose(z_eager, z_compiled)
    print("compiled mode OK:", z_compiled.shape)

if __name__ == "__main__":
    main()