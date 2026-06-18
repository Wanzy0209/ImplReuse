import torch
import tensorflow as tf
import numpy as np

class M(tf.Module):
    def __init__(self):
        super().__init__()
        # Introduce parameter for broadcasting / stride ops
        self.p = tf.Variable(2.0, dtype=tf.float32)

    @tf.function
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # x mimics STFT output (complex tensor)
        # Split into real/imag parts
        R = tf.abs(tf.math.real(x))   # unary op on real
        I = tf.math.imag(x) / self.p  # scalar divide with Variable (stride/broadcast)

        # Recombine using tf.math.add (candidate for reuse)
        # This mirrors the logic of torch.complex(R, I) by using addition of complex components
        R_complex = tf.cast(R, tf.complex64)
        I_complex = tf.cast(I, tf.complex64)
        Z = tf.math.add(R_complex, I_complex * 1j)
        return Z

def main():
    # Setup input mimicking STFT output (Batch, Freq, Time)
    # Using random complex numbers to simulate the output of torch.stft
    x = tf.complex(tf.random.normal((1, 257, 100)), tf.random.normal((1, 257, 100)))
    m = M()

    # Eager execution
    tf.config.run_functions_eagerly(True)
    z_eager = m(x)
    assert z_eager.dtype == tf.complex64
    print("eager mode OK:", z_eager.shape, z_eager.dtype)

    # Compiled execution (tf.function uses XLA/Graph compiler)
    tf.config.run_functions_eagerly(False)
    z_compiled = m(x)
    assert z_compiled.dtype == tf.complex64
    print("compiled mode OK:", z_compiled.shape, z_compiled.dtype)

    # Verify consistency between eager and compiled modes
    np.testing.assert_allclose(z_eager.numpy(), z_compiled.numpy())

if __name__ == "__main__":
    main()