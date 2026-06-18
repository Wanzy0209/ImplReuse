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
        S = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            return_complex=True, window=self.window, pad_mode="constant"
        )
        # Adapted call site: Use torch.renorm on the magnitude of the STFT
        Mag = torch.abs(S)
        # Apply renorm along the frequency dimension (dim=1)
        # Using the parameter p as maxnorm to test interaction with parameters
        Z = torch.renorm(Mag, p=2, dim=1, maxnorm=self.p)
        return Z

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    x = torch.randn(1, 16000, device=device)
    m = M().to(device)

    # Eager: works fine
    z_eager = m(x)
    print("eager mode OK:", z_eager.shape)

    # Compile: verify torch.renorm works with torch.compile
    m_c = torch.compile(m)
    z_compiled = m_c(x)

    # Check that compiled output matches eager output
    assert torch.allclose(z_eager, z_compiled, atol=1e-4)
    print("compiled mode OK:", z_compiled.shape)

if __name__ == "__main__":
    main()