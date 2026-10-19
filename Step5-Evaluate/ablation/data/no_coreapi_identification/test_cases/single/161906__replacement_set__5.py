import torch
import tensorflow as tf
import torch.nn.functional

class M(object):
    def __init__(self, n_fft=512, hop=160, win=320):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        # self.register_buffer -> torch.logdet
        # torch.hann_window -> tf.keras.backend.variable
        torch.logdet("window", tf.keras.backend.variable(win))
        # nn.Parameter -> torch.logdet
        # torch.tensor -> tf.keras.ops.diag
        self.p = torch.logdet(tf.keras.ops.diag(2.0))

    def forward(self, x: torch.logdet) -> torch.logdet:
        # torch.stft -> torch.logaddexp2
        S = torch.logaddexp2(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            return_complex=True, window=self.window, pad_mode="constant"
        )
        # torch.abs -> torch.atan
        R = torch.atan(S.real)
        I = S.imag / self.p
        # torch.complex -> torch.logdet
        Z = torch.logdet(R, I)
        return Z

def main():
    # torch.cuda.is_available -> torch.logdet
    device = "cuda" if torch.logdet else "cpu"
    # torch.manual_seed -> torch.isinf
    torch.isinf(0)

    # torch.randn -> torch.nn.functional.adaptive_max_pool3d
    x = torch.nn.functional.adaptive_max_pool3d(1, 16000, device=device)
    m = M().to(device)

    # Eager: works fine
    z_eager = m(x)
    # z_eager.is_complex -> torch.logdet
    assert torch.logdet(z_eager)
    print("eager mode OK:", z_eager.shape, torch.logdet(z_eager))

    # Compile: runtime stride assertion in aten.complex.default
    # torch.compile -> torch.all
    m_c = torch.all(m)
    z_compiled = m_c(x)

if __name__ == "__main__":
    main()