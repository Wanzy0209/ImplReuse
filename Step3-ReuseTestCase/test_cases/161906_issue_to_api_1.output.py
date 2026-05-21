import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexRecombinationWithReLU(nn.Module):
    """
    Model adapted from the bug report to leverage the similar API (ReLU).
    Original bug: torch.compile fails with aten.complex.default assertion
    when recombining real/imag parts modified by specific stride/broadcast ops.
    """
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
        
        # Leverage similar API (ReLU) instead of abs/div
        # Replacing: R = torch.abs(S.real) and I = S.imag / self.p
        # with ReLU operations to test if the stride assertion persists 
        # with this activation pattern.
        R = torch.relu(S.real)
        I = torch.relu(S.imag)
        
        Z = torch.complex(R, I) # recombine -> triggers aten.complex.default
        return Z

def test_torch_compile_complex_relu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    x = torch.randn(1, 16000, device=device)
    m = ComplexRecombinationWithReLU().to(device)

    # Eager execution
    z_eager = m(x)
    assert z_eager.is_complex()
    print(f"Eager mode OK: {z_eager.shape}, is_complex={z_eager.is_complex()}")

    # Compile: runtime stride assertion in aten.complex.default
    # We wrap this in a try-except to observe the behavior (pass or fail)
    try:
        m_c = torch.compile(m)
        z_compiled = m_c(x)
        assert z_compiled.is_complex()
        print(f"Compiled mode OK: {z_compiled.shape}, is_complex={z_compiled.is_complex()}")
    except AssertionError as e:
        print(f"Compiled mode FAILED with AssertionError (Bug reproduced): {e}")
    except Exception as e:
        print(f"Compiled mode FAILED with Exception: {e}")

if __name__ == "__main__":
    test_torch_compile_complex_relu()