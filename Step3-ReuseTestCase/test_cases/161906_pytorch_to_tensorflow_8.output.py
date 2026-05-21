import torch
import tensorflow as tf
import numpy as np

class M(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Introduce parameter for broadcasting / stride ops
        # Mimics the PyTorch nn.Parameter usage
        self.p = tf.Variable(2.0, trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Mimic the operations in PyTorch forward pass that affect strides/shapes
        # PyTorch: R = torch.abs(S.real); I = S.imag / self.p
        # Here we perform a scalar divide with Parameter (broadcasting)
        # to ensure the tensor passed to the API has specific properties.
        y = x / self.p 
        
        # Call the target API: tf.keras.ops.tril
        # This replaces torch.complex(R, I) from the original bug report
        Z = tf.keras.ops.tril(y)
        return Z

def main():
    # Setup
    tf.random.set_seed(0)
    
    # Input tensor: Batch of matrices
    # PyTorch input was (1, 16000). We use (1, 5, 5) for tril.
    x = tf.random.normal((1, 5, 5))
    
    m = M()

    # Eager: works fine
    z_eager = m(x)
    print("eager mode OK:", z_eager.shape)

    # Compile: tf.function is the TensorFlow equivalent of torch.compile
    # It traces the graph and applies optimizations, potentially exposing
    # stride/broadcasting issues similar to the PyTorch Inductor backend.
    m_compiled = tf.function(m)
    
    try:
        z_compiled = m_compiled(x)
        print("compiled mode OK:", z_compiled.shape)
        
        # Verify results match between eager and compiled execution
        assert np.allclose(z_eager.numpy(), z_compiled.numpy())
        print("Results match!")
    except Exception as e:
        print(f"Compiled execution error: {e}")

if __name__ == "__main__":
    main()