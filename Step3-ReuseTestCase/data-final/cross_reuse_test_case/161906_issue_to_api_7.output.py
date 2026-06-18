import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np

def test_tf_experimental_numpy_add_complex_parts():
    """
    Test case for tf.experimental.numpy.add based on the logic of PyTorch issue 161906.
    
    The original issue involves torch.compile failing on torch.complex when inputs
    are derived from complex tensor parts (real/imag) with specific operations (abs, div).
    
    This test adapts that logic to verify that tf.experimental.numpy.add handles
    similarly derived inputs (real and imaginary parts modified by unary/binary ops)
    correctly, especially under tf.function (TensorFlow's compilation equivalent).
    """
    
    # Setup: Create a complex tensor mimicking STFT output
    # Shape: (Batch, Freq, Time)
    batch, freq, time = 1, 257, 100
    real_data = np.random.randn(batch, freq, time).astype(np.float32)
    imag_data = np.random.randn(batch, freq, time).astype(np.float32)
    
    # S is the complex input
    S = tf.complex(real_data, imag_data)
    
    # Parameter for scalar division (mimicking nn.Parameter in the original bug)
    p = tf.Variable(2.0)

    # Define the computation graph (mimicking torch.compile)
    @tf.function
    def compute(x):
        # Extract real and imaginary parts
        R = tf.math.real(x)
        I = tf.math.imag(x)

        # Apply operations that might affect strides/views in eager/compiled modes
        # Corresponds to: R = torch.abs(S.real)
        R_mod = tf.abs(R)

        # Corresponds to: I = S.imag / self.p
        # This introduces a broadcast operation
        I_mod = I / p

        # Recombine using the Similar API: tf.experimental.numpy.add
        # Original bug trigger: Z = torch.complex(R, I)
        # Here we use tnp.add to combine the modified real and imaginary parts.
        # Note: Mathematically this is R + I, not R + i*I, but structurally it 
        # replaces the binary op combining the two derived tensors.
        Z = tnp.add(R_mod, I_mod)
        return Z

    # Execute
    try:
        result = compute(S)
        
        # Assertions
        assert result.shape == (batch, freq, time), f"Shape mismatch: {result.shape}"
        # Since we added two real tensors, the result should be real
        assert result.dtype == tf.float32, f"Dtype mismatch: {result.dtype}"
        
        print("Test passed: tf.experimental.numpy.add handled modified real/imag parts correctly.")
        print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_tf_experimental_numpy_add_complex_parts()