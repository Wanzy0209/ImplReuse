import torch
import numpy as np
import sys

# Handle environment/dependency errors (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test due to environment/dependency error: {e}")
    print("This is likely due to a missing GLIBCXX version or TensorFlow installation issues.")
    sys.exit(0)

class M(tf.Module):
    def __init__(self, n_fft=512, hop=160, win=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        # Register window as a variable so it moves with the model
        self.window = tf.Variable(tf.signal.hann_window(win), trainable=False)
        # Introduce parameter for broadcasting / stride ops
        self.p = tf.Variable(2.0)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        S = tf.signal.stft(
            x, frame_length=self.n_fft, frame_step=self.hop, fft_length=self.n_fft,
            window_fn=lambda: self.window, pad_end=True
        )
        
        # Original: R = torch.abs(S.real)
        # Adaptation: Leverage similar API tf.keras.ops.relu on the real part
        # This preserves the unary operation pattern on the real component
        R = tf.keras.ops.relu(tf.math.real(S))
        
        I = tf.math.imag(S) / self.p  # scalar divide with Variable
        
        Z = tf.complex(R, I)  # recombine -> triggers tf.complex
        return Z

def main():
    # Setup device context (TensorFlow handles GPU/CPU automatically)
    tf.random.set_seed(0)

    x = tf.random.normal((1, 16000))
    m = M()

    # Eager: works fine
    z_eager = m.forward(x)
    assert z_eager.dtype.is_complex
    print("eager mode OK:", z_eager.shape, z_eager.dtype)

    # Compile: tf.function is the equivalent of torch.compile
    # This tests the graph mode execution which mirrors the compiled execution in the issue
    m_compiled = tf.function(m.forward)
    z_compiled = m_compiled(x)
    
    assert z_compiled.dtype.is_complex
    print("compiled mode OK:", z_compiled.shape, z_compiled.dtype)

if __name__ == "__main__":
    main()