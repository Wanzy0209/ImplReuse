import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, n_fft=1024, hop=512, win=1024):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        # Register window as buffer so it moves with .to(device)
        self.register_buffer("window", torch.hann_window(win))
        # Introduce parameter for broadcasting / stride ops
        self.p = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            return_complex=True, window=self.window, pad_mode="constant"
        )
        R = torch.abs(S.real)   # unary op on real
        I = S.imag / self.p     # scalar divide with Parameter (stride/broadcast)
        
        # Original call: Z = torch.complex(R, I)
        # Adapted call: Use torch.randint_like with R as the reference tensor
        # This tests if the compiler handles size/stride inference correctly for the similar API
        Z = torch.randint_like(R, high=100, dtype=torch.int32)
        return Z

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Adjusted input size to be compatible with hop/n_fft (power of 2)
    x = torch.randn(1, 16384, device=device)
    m = M().to(device)

    # Eager: works fine
    z_eager = m(x)
    assert z_eager.dtype == torch.int32
    print("eager mode OK:", z_eager.shape, z_eager.dtype)

    # Compile: verify no stride assertion errors occur with torch.randint_like
    m_c = torch.compile(m)
    z_compiled = m_c(x)

    # Verify results match shape and type
    assert z_compiled.shape == z_eager.shape
    assert z_compiled.dtype == torch.int32
    print("compiled mode OK:", z_compiled.shape, z_compiled.dtype)

if __name__ == "__main__":
    main()