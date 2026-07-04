import torch
import time

# Attempt to import TensorFlow, handling potential environment dependency errors
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Test skipped: Cannot import TensorFlow due to environment issues.")
    print(f"Error details: {e}")
    print("This is often caused by missing system libraries (e.g., GLIBCXX).")
    import sys
    sys.exit(0)

# Shapes extracted from the original torch.matmul regression bug report
shapes = [
    ((1, 12, 10, 64), (1, 12, 64, 10)),
    ((1, 12, 10, 10), (1, 12, 10, 64)),
]

def test_matmul_with_control_dependencies():
    """
    Test case adapted from torch.matmul regression (Issue 162683).
    Preserves the tensor shapes and benchmarking logic but adapts
    the execution to use tf.control_dependencies to ensure ordering,
    reflecting the similar API's usage pattern.
    """
    for a_shape, b_shape in shapes:
        # Setup tensors similar to the original bug (float16, uniform distribution)
        A = tf.random.uniform(a_shape, minval=-1, maxval=1, dtype=tf.float16)
        B = tf.random.uniform(b_shape, minval=-1, maxval=1, dtype=tf.float16)
        
        # Use a dummy variable to create a control dependency context.
        # This leverages the 'tf.control_dependencies' API to manage execution flow,
        # similar to how the original report managed threading/execution environment.
        dummy_var = tf.Variable(0.0, dtype=tf.float16)

        @tf.function
        def controlled_matmul():
            # Ensure the dummy operation (e.g., a state update) completes before matmul
            with tf.control_dependencies([dummy_var.assign(1.0)]):
                return tf.matmul(A, B)

        # Warmup (preserving original logic)
        for _ in range(10):
            _ = controlled_matmul()

        # Run and verify (preserving original loop structure but adding assertions)
        times = []
        repeat = 50 
        for i in range(repeat):
            start = time.time()
            result = controlled_matmul()
            end = time.time()
            
            # Verify shape correctness to ensure the operation ran as expected
            # The last dimension of A must match the second to last of B for batch matmul
            expected_shape = tf.TensorShape([a_shape[0], a_shape[1], a_shape[2], b_shape[3]])
            assert result.shape == expected_shape, \
                f"Shape mismatch for {a_shape} x {b_shape}: got {result.shape}, expected {expected_shape}"
            
            if i > 10:
                times.append((end - start) * 1000)

        if times:
            avg_time = sum(times) / len(times)
            print(f"Test {a_shape} x {b_shape} -> Avg: {avg_time:.3f} ms")

if __name__ == "__main__":
    test_matmul_with_control_dependencies()