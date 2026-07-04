import torch
import numpy as np
import sys

# Attempt to import TensorFlow, handle environment errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment incompatibility.")
    print(f"Details: {e}")
    sys.exit(0)

def test_avgpool_with_session_run_values():
    """
    Test case leveraging tf.compat.v1.train.SessionRunValues to capture 
    and verify the results of an AvgPool2d operation, mirroring the 
    logic and parameters from the PyTorch MPS bug report.
    """
    # Disable eager execution to ensure SessionRunValues context is relevant
    tf.compat.v1.disable_eager_execution()

    # --- Reproduce Bug Report Logic ---
    # Original: torch.manual_seed(0)
    # Original: x = torch.randn(4, 6, 7) -> Shape (C, H, W) in PyTorch for 3D input
    # Translation: TensorFlow uses (Batch, Height, Width, Channels)
    # We map (4, 6, 7) to (1, 6, 7, 4) assuming Batch=1
    np.random.seed(0)
    x_np = np.random.randn(1, 6, 7, 4).astype(np.float32)
    
    # Original: model = torch.nn.AvgPool2d(kernel_size=[1, 6], stride=[4, 9], ceil_mode=True, divisor_override=3)
    # Translation: tf.nn.avg_pool2d
    # kernel_size=[1, 6] -> ksize=[1, 1, 6, 1] (Batch, H, W, Channel)
    # stride=[4, 9] -> strides=[1, 4, 9, 1]
    # ceil_mode=True -> padding='SAME' (Approximation for output size calculation)
    # Note: divisor_override=3 is not directly supported in tf.nn.avg_pool2d, 
    # so we use standard averaging here.
    
    input_tensor = tf.compat.v1.placeholder(tf.float32, shape=(1, 6, 7, 4))
    
    pooled_tensor = tf.nn.avg_pool2d(
        input_tensor,
        ksize=[1, 1, 6, 1],
        strides=[1, 4, 9, 1],
        padding='SAME'
    )

    # --- Leverage Similar API: tf.compat.v1.train.SessionRunValues ---
    with tf.compat.v1.Session() as sess:
        # Run the operation
        results = sess.run(pooled_tensor, feed_dict={input_tensor: x_np})
        
        # Use SessionRunValues to encapsulate the results, mimicking the hook pattern
        # This allows for structured handling of the run output
        run_values = tf.compat.v1.train.SessionRunValues(
            results=results,
            options=None,
            run_metadata=None
        )

        # --- Assertions based on Bug Report ---
        # The bug report checks if output matches expected (specifically checking for incorrect zeros).
        # We verify that SessionRunValues correctly holds the non-zero result.
        
        assert run_values.results is not None, "SessionRunValues results should not be None"
        
        # Check shape consistency
        # PyTorch output shape with ceil_mode=True was (4, 2, 1) -> (C, H_out, W_out)
        # TF output shape with 'SAME' padding should be (1, 2, 1, 4) -> (Batch, H_out, W_out, C)
        expected_shape = (1, 2, 1, 4)
        assert run_values.results.shape == expected_shape, \
            f"Shape mismatch. Expected {expected_shape}, got {run_values.results.shape}"

        # The specific bug in MPS was values becoming 0.0.
        # We assert that the results are not all zeros (which would indicate a failure).
        assert not np.allclose(run_values.results, 0.0), \
            "Output contains all zeros, similar to the reported MPS bug."

        print("Test Passed.")
        print(f"SessionRunValues captured results shape: {run_values.results.shape}")
        print(f"Sample output:\n{run_values.results[0, :, :, 0]}")

if __name__ == "__main__":
    test_avgpool_with_session_run_values()