```python
import tensorflow as tf

class M(tf.Module):
    def __init__(self, n_fft=512, hop=160, win=320):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        # Register window as buffer so it moves with .to(device)
        # In TensorFlow, tensors assigned to self are tracked automatically.
        self.window = tf.signal.hann_window(self.win)
        # Introduce parameter for broadcasting / stride ops
        # nn.Parameter maps to tf.Variable
        self.p = tf.Variable(2.0, dtype=tf.float32)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Note: PyTorch stft returns (batch, freq, time), TF returns (batch, time, freq).
        # PyTorch returns one-sided spectrum for real inputs, TF returns two-sided (fft_length).
        S = tf.signal.stft(
            x, 
            frame_length=self.win, 
            frame_step=self.hop, 
            fft_length=self.n_fft,
            window_fn=lambda f: self.window, # Pass pre-computed window
            pad_end=True, 
            pad_mode='CONSTANT' # Matches PyTorch "constant"
        )
        R = tf.abs(tf.math.real(S))   # unary op on real
        I = tf.math.imag(S) / self.p     # scalar divide with Variable (stride/broadcast)
        Z = tf.complex(R, I) # recombine -> triggers complex op
        return Z

def main():
    # TensorFlow handles device placement automatically (GPU if available)
    device = "GPU" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU"
    tf.random.set_seed(0)

    x = tf.random.normal((1, 16000))
    m = M()

    # Eager: works fine
    z_eager = m(x)
    assert z_eager.dtype.is_complex
    print("eager mode OK:", z_eager.shape, z_eager.dtype.is_complex)

    # Compile: tf.function is the TensorFlow equivalent for graph compilation
    m_c = tf.function(m)
    z_compiled = m_c(x)

if __name__ == "__main__":
    main()
```