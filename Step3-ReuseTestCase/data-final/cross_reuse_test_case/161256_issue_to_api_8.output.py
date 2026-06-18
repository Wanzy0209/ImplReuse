import torch
import tensorflow as tf

def test_float64_matmul_with_summary_step():
    """
    Test case adapted from PyTorch Issue #161256.
    Reproduces a float64 matrix multiplication crash on ROCm.
    Leverages the similar API 'tf.summary.experimental.set_step' to mirror
    the state-setting pattern of 'torch.set_default_dtype' found in the original issue.
    """
    
    # Leverage the similar API: tf.summary.experimental.set_step
    # This mirrors the 'torch.set_default_dtype' call in the original bug report,
    # acting as a state-setting operation before the computation.
    tf.summary.experimental.set_step(1)

    # Check for GPU availability to match the original "device='cuda'" context
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Test requires a GPU (ROCm) to reproduce the specific crash context.")
        return

    try:
        # Force placement on GPU
        with tf.device('/GPU:0'):
            # Original logic: Create float64 tensors
            # PyTorch: torch.set_default_dtype(torch.float64); x = torch.randn(...)
            # TensorFlow: Explicitly set dtype=tf.float64
            x = tf.random.normal((1000, 1000), dtype=tf.float64)
            y = tf.random.normal((1000, 1000), dtype=tf.float64)

            # Original logic: Perform Matmul
            # PyTorch: z = x @ y
            # TensorFlow: z = tf.linalg.matmul(x, y)
            z = tf.linalg.matmul(x, y)

            # Original logic: Access element to trigger execution
            # PyTorch: print(z[0, 0])
            result = z[0, 0]
            
            # If running on ROCm 6.4.3 with affected TF versions, this may segfault here.
            print(f"Computation successful. Result[0,0]: {result.numpy()}")
            
    except Exception as e:
        print(f"Exception occurred during matmul: {e}")

if __name__ == "__main__":
    test_float64_matmul_with_summary_step()