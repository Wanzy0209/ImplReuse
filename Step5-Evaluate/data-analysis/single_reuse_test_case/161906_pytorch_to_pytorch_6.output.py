import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, n_fft=512, hop=160, win=320):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        # Register window as buffer so it moves with .to(device)
        self.register_buffer("window", torch.hann_window(win))
        # Introduce parameter for broadcasting / stride ops
        self.p = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix: Ensure input is contiguous to avoid cuFFT internal errors
        x = x.contiguous()
        S = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            return_complex=True, window=self.window, center=False
        )
        R = torch.abs(S.real)   # unary op on real
        I = S.imag / self.p     # scalar divide with Parameter (stride/broadcast)
        
        # Adaptation: Replace torch.complex(R, I) with torch.arange
        # We use torch.arange to generate a tensor of the same shape as R
        # to verify torch.arange works correctly under torch.compile in this context.
        Z = torch.arange(0, R.numel(), device=R.device, dtype=R.dtype).reshape(R.shape)
        return Z

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    x = torch.randn(1, 16000, device=device)
    m = M().to(device)

    # Eager: works fine
    z_eager = m(x)
    print("eager mode OK:", z_eager.shape)

    # Compile: verify torch.arange works without stride assertion errors
    m_c = torch.compile(m)
    z_compiled = m_c(x)
    print("compiled mode OK:", z_compiled.shape)

    # Verify consistency
    assert torch.allclose(z_eager, z_compiled)

if __name__ == "__main__":
    main()