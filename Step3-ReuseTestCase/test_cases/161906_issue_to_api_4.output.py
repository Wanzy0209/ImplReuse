import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        # Register window as buffer so it moves with .to(device)
        self.register_buffer("window", torch.hann_window(16))
        # Introduce parameter for broadcasting / stride ops
        self.p = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use STFT to generate a complex tensor, similar to the original bug context
        S = torch.stft(
            x, n_fft=16, hop_length=4, win_length=16,
            return_complex=True, window=self.window, pad_mode="constant"
        )
        
        # Slice the STFT output to create square matrices (Batch, 4, 4)
        # This preserves the memory layout characteristics from STFT
        S_sq = S[:, :4, :4]
        
        # Apply scalar division with Parameter (stride/broadcast trigger)
        # This mirrors the 'I = S.imag / self.p' line in the original bug report
        S_mod = S_sq / self.p
        
        # Call the similar API: torch.logdet
        # This replaces 'torch.complex(R, I)' from the original bug
        return torch.logdet(S_mod)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Input length chosen to ensure enough frames for slicing
    x = torch.randn(2, 160, device=device)
    m = M().to(device)

    # Eager execution
    z_eager = m(x)
    print("eager mode OK:", z_eager.shape)

    # Compiled execution
    # The original bug failed here with AssertionError for aten.complex.default
    # We test if torch.logdet (similar API) handles the strides correctly under compile
    m_c = torch.compile(m)
    z_compiled = m_c(x)
    
    # Verify results match
    assert torch.allclose(z_eager, z_compiled)
    print("compiled mode OK:", z_compiled.shape)

if __name__ == "__main__":
    main()