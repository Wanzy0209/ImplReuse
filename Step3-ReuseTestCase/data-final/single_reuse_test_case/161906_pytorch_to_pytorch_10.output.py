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
        # Adaptation: Use torch.movedim to manipulate strides of the complex tensor
        # instead of recombining real/imag parts via torch.complex.
        # This verifies that torch.compile handles stride changes correctly.
        Z = torch.movedim(S, 0, 1)
        return Z

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    x = torch.randn(1, 16000, device=device)
    m = M().to(device)

    # Eager: works fine
    z_eager = m(x)
    assert z_eager.is_complex()
    print("eager mode OK:", z_eager.shape, z_eager.is_complex())

    # Compile: verify torch.movedim works correctly with torch.compile
    m_c = torch.compile(m)
    z_compiled = m_c(x)

    # Check if results match
    assert torch.allclose(z_eager, z_compiled)
    print("compiled mode OK:", z_compiled.shape, z_compiled.is_complex())

if __name__ == "__main__":
    main()