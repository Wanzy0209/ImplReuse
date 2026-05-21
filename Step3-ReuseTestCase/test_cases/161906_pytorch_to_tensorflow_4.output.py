import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

class M(tf.Module):
    def __init__(self):
        super().__init__()
        # Introduce parameter for broadcasting / stride ops
        self.p = tf.Variable(2.0, dtype=tf.float32)

    def forward_eager(self, shape):
        # Use the similar API (tf.experimental.numpy.random.uniform) 
        # to generate Real and Imaginary parts, mimicking the output of torch.stft
        R = tnp.random.uniform(low=-1.0, high=1.0, size=shape)
        I = tnp.random.uniform(low=-1.0, high=1.0, size=shape)

        R = tf.abs(R)   # unary op on real
        I = I / self.p  # scalar divide with Variable (stride/broadcast)
        Z = tf.complex(R, I) # recombine -> triggers tf.complex
        return Z

    @tf.function # Equivalent to torch.compile
    def forward_compiled(self, shape):
        # Use the similar API to generate Real and Imaginary parts
        R = tnp.random.uniform(low=-1.0, high=1.0, size=shape)
        I = tnp.random.uniform(low=-1.0, high=1.0, size=shape)

        R = tf.abs(R)
        I = I / self.p
        Z = tf.complex(R, I)
        return Z

def main():
    tf.random.set_seed(0)
    
    # Shape mimicking STFT output (batch, freq, time)
    shape = (1, 257, 100) 

    m = M()

    # Eager: works fine
    z_eager = m.forward_eager(shape)
    assert z_eager.dtype == tf.complex64
    print("eager mode OK:", z_eager.shape, z_eager.dtype)

    # Compiled (Graph mode): verify no assertion errors
    z_compiled = m.forward_compiled(shape)
    assert z_compiled.dtype == tf.complex64
    print("compiled mode OK:", z_compiled.shape, z_compiled.dtype)

if __name__ == "__main__":
    main()