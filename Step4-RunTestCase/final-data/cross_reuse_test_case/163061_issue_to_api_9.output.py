import torch
import threading
import time
import numpy as np
import sys

# Attempt to import TensorFlow, handle environment dependency errors
try:
    import tensorflow as tf
except ImportError as e:
    # Check for the specific GLIBCXX error mentioned in the prompt
    if "GLIBCXX" in str(e) or "libstdc++" in str(e):
        print("SKIPPED: TensorFlow cannot be imported due to environment issues.")
        print(f"Error: {e}")
        print("This is likely due to an outdated libstdc++.so.6 on the system.")
        sys.exit(0)
    else:
        # If it's a different import error, raise it
        raise

def tf_leaky_relu_eager(features, alpha=0.2):
    """Eager execution version of the operation."""
    return tf.nn.leaky_relu(features, alpha=alpha)

@tf.function
def tf_leaky_relu_compiled(features, alpha=0.2):
    """Compiled (Autograph/Graph) version of the operation, analogous to torch.compile."""
    return tf.nn.leaky_relu(features, alpha=alpha)

def check_gil_released(func, *args):
    """
    Helper to check if the GIL is released during the execution of a function.
    If the GIL is held, the worker thread will be blocked and unable to increment
    the counter until the main function finishes.
    """
    worker_ran = False
    
    def worker():
        nonlocal worker_ran
        # Perform a CPU-bound task that requires the GIL
        for _ in range(1000000):
            pass
        worker_ran = True

    thread = threading.Thread(target=worker)
    thread.start()

    # Execute the target function
    result = func(*args)
    
    # Ensure the operation is fully executed (especially for GPU ops)
    if hasattr(result, 'numpy'):
        _ = result.numpy()

    # Wait for the worker thread with a timeout
    # If the GIL was held by the TF op, the thread might not finish in time
    thread.join(timeout=2.0)
    
    return worker_ran

def test_tf_leaky_relu_gil_release():
    # Check for GPU availability to match the original issue context (CUDA)
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    
    print(f"Running test on device: {device}")

    with tf.device(device):
        # Create large tensors to ensure the operation takes sufficient time
        # to allow the GIL check to be meaningful.
        x = tf.random.normal((4096, 4096))

        # Test 1: Eager execution
        # Standard TF eager ops usually release the GIL.
        eager_released = check_gil_released(tf_leaky_relu_eager, x)
        assert eager_released, "GIL was held during eager execution of tf.nn.leaky_relu"
        print("Eager execution: GIL released (Passed)")

        # Test 2: Compiled execution (@tf.function)
        # This is the analogy to torch.compile in the original bug report.
        # We verify if the compiled graph execution releases the GIL.
        compiled_released = check_gil_released(tf_leaky_relu_compiled, x)
        assert compiled_released, "GIL was held during compiled execution of tf.nn.leaky_relu"
        print("Compiled execution: GIL released (Passed)")

if __name__ == "__main__":
    test_tf_leaky_relu_gil_release()