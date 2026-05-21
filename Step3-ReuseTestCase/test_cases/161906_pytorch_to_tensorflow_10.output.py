import torch
import tensorflow as tf
import numpy as np

class TriuModel(tf.Module):
    def __init__(self, n_fft=512, hop=160, win=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        # Register window as a variable/buffer so it moves with .to(device) equivalent
        self.window = tf.signal.hann_window(win)
        # Introduce parameter for broadcasting / stride ops
        self.p = tf.Variable(2.0, dtype=tf.float32)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # TensorFlow STFT equivalent
        # Note: tf.signal.stft returns shape (batch, time, freq)
        S = tf.signal.stft(
            x, 
            frame_length=self.n_fft, 
            frame_step=self.hop, 
            fft_length=self.n_fft,
            window_fn=lambda _: self.window,
            pad_end=True
        )
        
        # Mimic the logic from the PyTorch bug:
        # Operations that might affect strides/broadcasting
        R = tf.abs(tf.math.real(S))   # unary op on real
        I = tf.math.imag(S) / self.p  # scalar divide with Variable (stride/broadcast)
        
        # Reccombine (PyTorch bug location: torch.complex)
        Z = tf.complex(R, I)
        
        # Target API: tf.experimental.numpy.triu
        # This replaces torch.complex as the API under test
        return tf.experimental.numpy.triu(Z)

def main():
    # Device setup
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    
    with tf.device(device_name):
        tf.random.set_seed(0)

        x = tf.random.normal((1, 16000))
        m = TriuModel()

        # Eager: works fine
        print("Testing Eager mode...")
        z_eager = m(x)
        # Check if complex (tf.experimental.numpy.triu preserves dtype)
        assert z_eager.dtype.is_complex
        print("eager mode OK:", z_eager.shape, z_eager.dtype.is_complex)

        # Compile: runtime stride assertion check
        # Using tf.function to simulate torch.compile
        print("Testing Compiled mode (tf.function)...")
        m_c = tf.function(m)
        z_compiled = m_c(x)
        
        # Verify outputs match
        np.testing.assert_allclose(z_eager.numpy(), z_compiled.numpy())
        print("compiled mode OK:", z_compiled.shape, z_compiled.dtype.is_complex)

if __name__ == "__main__":
    main()