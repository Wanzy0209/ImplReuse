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
        # Leverage the similar API (tf.nn.relu -> torch.relu) 
        # instead of torch.abs to test the stride assertion logic
        R = torch.relu(S.real)   
        I = S.imag / self.p     # scalar divide with Parameter (stride/broadcast)
        Z = torch.complex(R, I) # recombine -> triggers aten.complex.default
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

    # Compile: runtime stride assertion in aten.complex.default
    m_c = torch.compile(m)
    z_compiled = m_c(x)
    
    # Verify consistency
    assert torch.allclose(z_eager, z_compiled, atol=1e-4)
    print("compiled mode OK:", z_compiled.shape, z_compiled.is_complex())

if __name__ == "__main__":
    main()