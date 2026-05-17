import torch
import tensorflow as tf

def test_tf_keras_random_uniform_dynamic_shapes():
    """
    Adapted test case based on PyTorch Issue 160882.
    Original Bug: torch.compile crashes when torch.complex receives inputs 
    with different shapes than during compilation.
    
    Target API: tf.keras.random.uniform
    Logic: Verify if the target API handles dynamic shape changes correctly 
    when traced with tf.function (TensorFlow's compilation equivalent).
    """
    
    # Define the function using the target API
    # Original: z = torch.complex(real, imag)
    # Target: z = tf.keras.random.uniform(low, high)
    @tf.function
    def f(low: tf.Tensor, high: tf.Tensor) -> tf.Tensor:
        return tf.keras.random.uniform(low, high)

    B, F, T = 1, 641, 39

    # Create source tensors
    # Using constants to ensure deterministic behavior for the test case
    low_src = tf.ones((B, F, T))
    high_src = tf.ones((B, F, T)) * 2.0

    # Create mismatched tensors (permuted dimensions)
    # Original: r_mismatch = r_src.permute(0, 2, 1)
    low_mismatch = tf.transpose(low_src, [0, 2, 1])
    high_mismatch = tf.transpose(high_src, [0, 2, 1])

    # First call (Compilation/Tracing)
    # Original: _ = compiled(r_src, i_src)
    try:
        out1 = f(low_src, high_src)
        assert out1.shape == (B, F, T), f"First call failed: Expected shape {(B, F, T)}, got {out1.shape}"
        print("First call successful. Shape:", out1.shape)
    except Exception as e:
        print(f"First call failed: {e}")
        raise

    # Second call with different shapes
    # Original: _ = compiled(r_mismatch, i_mismatch)
    # This tests if the compiled/traced function handles the shape change
    # similar to the bug report scenario.
    try:
        out2 = f(low_mismatch, high_mismatch)
        assert out2.shape == (B, T, F), f"Second call failed: Expected shape {(B, T, F)}, got {out2.shape}"
        print("Second call successful. Shape:", out2.shape)
    except Exception as e:
        print(f"Second call failed (Bug reproduced?): {e}")
        raise

if __name__ == "__main__":
    test_tf_keras_random_uniform_dynamic_shapes()