import torch
import tensorflow as tf
import numpy as np

class M(tf.keras.Model):
    def __init__(self, n_fft=512, hop=160, win=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        # Register window as buffer/variable so it moves with .to(device) equivalent
        self.window = tf.signal.hann_window(win)
        # Introduce parameter for broadcasting / stride ops
        self.p = tf.Variable(2.0, trainable=True)

    def call(self, x, training=False):
        # STFT equivalent
        S = tf.signal.stft(
            x, frame_length=self.n_fft, frame_step=self.hop, fft_length=self.n_fft,
            window_fn=lambda: self.window, pad_end=True
        )
        
        R = tf.abs(tf.math.real(S))   # unary op on real
        I = tf.math.imag(S) / self.p  # scalar divide with Variable (stride/broadcast)
        
        # Adaptation: Apply the Similar API (tf.keras.random.dropout)
        # We apply it to the real part R to test the API in this context.
        # Note: dropout behavior differs between training and inference.
        if training:
            # Using a fixed seed for reproducibility in the test case
            R = tf.keras.random.dropout(R, rate=0.5, seed=42)
            
        # Recombine into complex tensor to match original output structure
        Z = tf.complex(R, I)
        return Z

def main():
    # Setup device (TensorFlow handles GPU/CPU automatically)
    tf.random.set_seed(0)

    x = tf.random.normal((1, 16000))
    m = M()

    # Eager: works fine
    z_eager = m(x, training=True)
    assert z_eager.dtype == tf.complex64
    print("eager mode OK:", z_eager.shape, z_eager.dtype)

    # Compiled: runtime check (tf.function is the equivalent of torch.compile)
    m_compiled = tf.function(m)
    z_compiled = m_compiled(x, training=True)
    
    assert z_compiled.dtype == tf.complex64
    print("compiled mode OK:", z_compiled.shape, z_compiled.dtype)

if __name__ == "__main__":
    main()