import torch
import tensorflow as tf
import numpy as np

class M(tf.Module):
    def __init__(self):
        super().__init__()
        # Introduce parameter for broadcasting / stride ops
        self.p = tf.Variable(2.0, dtype=tf.float32)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        # Original: S = torch.stft(...)
        # Adaptation: Assume x is a matrix suitable for tril operations
        
        # Original: R = torch.abs(S.real)   # unary op on real
        R = tf.abs(x)
        
        # Original: I = S.imag / self.p     # scalar divide with Parameter (stride/broadcast)
        I = x / self.p
        
        # Original: Z = torch.complex(R, I) # recombine -> triggers aten.complex.default
        # Adaptation: Z = tf.experimental.numpy.tril(I)
        # This tests if tril handles the broadcasted strides correctly under compilation
        Z = tf.experimental.numpy.tril(I)
        return Z

def main():
    # Setup
    # Original: x = torch.randn(1, 16000)
    # Adaptation: x needs to be at least 2D for tril. Let's use (1, 512, 512)
    x = tf.random.normal((1, 512, 512))
    m = M()

    # Eager: works fine
    z_eager = m.forward(x)
    # Verify basic properties
    assert z_eager.shape == x.shape
    # Verify tril property (upper triangle should be zero)
    # Extract a slice from the upper triangle to check
    assert tf.reduce_all(z_eager[:, :, 10:] == 0).numpy()
    print("eager mode OK:", z_eager.shape)

    # Compile: runtime stride assertion in aten.complex.default
    # Adaptation: Use tf.function (TensorFlow's compilation mechanism)
    m_compiled = tf.function(m.forward)
    z_compiled = m_compiled(x)
    
    assert z_compiled.shape == x.shape
    assert tf.reduce_all(z_compiled[:, :, 10:] == 0).numpy()
    print("compiled mode OK:", z_compiled.shape)

if __name__ == "__main__":
    main()