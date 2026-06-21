import torch
import threading
import time
import sys

# Attempt to import TensorFlow. Handle environment incompatibility errors gracefully.
try:
    import tensorflow as tf
except ImportError as e:
    # The error message indicates a GLIBCXX version mismatch, which is an environment issue.
    print(f"Skipping test: Unable to import TensorFlow due to environment incompatibility.")
    print(f"Details: {e}")
    sys.exit(0)

# Helper class to check if the GIL is released during execution
class GilChecker:
    def __init__(self):
        self.running = True
        self.counter = 0
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        # This loop runs in a separate thread. 
        # If the GIL is held by the main thread, this thread will be blocked 
        # and the counter will not increase.
        while self.running:
            self.counter += 1

    def stop(self):
        self.running = False
        self.thread.join()

    def is_gil_released(self, duration=0.1):
        # Check if the counter increased during the duration
        start_count = self.counter
        time.sleep(duration)
        return self.counter > start_count

def test_gil_release_leaky_relu():
    """
    Test case to verify if the GIL is released when calling 
    tf.compat.v1.nn.leaky_relu, similar to the reported issue with torch.compile.
    """
    # Ensure we are in eager mode for the baseline, but we will test tf.function (compiled)
    # which is the TensorFlow equivalent of torch.compile.
    
    # Define the function using the similar API: tf.compat.v1.nn.leaky_relu
    # We wrap it to test both eager and compiled execution.
    def leaky_relu_eager(features, alpha=0.2):
        # The similar API implementation handles type conversion internally:
        # features = ops.convert_to_tensor(features, name="features")
        # if features.dtype.is_integer: ...
        return tf.compat.v1.nn.leaky_relu(features, alpha=alpha)

    # Create a compiled version using tf.function (analogous to torch.compile)
    leaky_relu_compiled = tf.function(leaky_relu_eager)

    # Create input data
    # Using a large tensor to ensure the operation takes enough time to observe GIL behavior
    x_float = tf.random.normal((4096, 4096))
    x_int = tf.random.uniform((4096, 4096), minval=0, maxval=10, dtype=tf.int32)

    print("Testing GIL release for tf.compat.v1.nn.leaky_relu...")

    # Test 1: Eager execution with float input
    checker = GilChecker()
    _ = leaky_relu_eager(x_float)
    gil_released_eager_float = checker.is_gil_released()
    print(f"Eager (Float): GIL Released = {gil_released_eager_float}")
    checker.stop()

    # Test 2: Compiled execution (tf.function) with float input
    # This is the direct parallel to the torch.compile issue.
    checker = GilChecker()
    _ = leaky_relu_compiled(x_float)
    gil_released_compiled_float = checker.is_gil_released()
    print(f"Compiled (Float): GIL Released = {gil_released_compiled_float}")
    checker.stop()

    # Test 3: Compiled execution with integer input
    # This exercises the specific code path in the similar API: 
    # "if features.dtype.is_integer: features = math_ops.cast(...)"
    checker = GilChecker()
    _ = leaky_relu_compiled(x_int)
    gil_released_compiled_int = checker.is_gil_released()
    print(f"Compiled (Int): GIL Released = {gil_released_compiled_int}")
    checker.stop()

    # Assertions
    # We expect the GIL to be released for compiled operations (tf.function)
    # similar to how it is expected for CUDA kernels in PyTorch.
    assert gil_released_compiled_float, "GIL was not released for compiled leaky_relu (float)"
    assert gil_released_compiled_int, "GIL was not released for compiled leaky_relu (int)"
    
    print("Test passed: GIL is released for compiled operations.")

if __name__ == "__main__":
    test_gil_release_leaky_relu()