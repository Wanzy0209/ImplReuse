import torch
import tensorflow as tf
import numpy as np
import time

def test_lu_matrix_inverse_performance_amp_static():
    """
    Test case for tf.linalg.lu_matrix_inverse inspired by PyTorch Issue #161878.
    
    The original issue reported a performance regression in torch._inductor 
    (PyTorch's compiler) when running BERT with AMP (Automatic Mixed Precision),
    static shapes, and batch size 2 on CPU.
    
    This test adapts that scenario to the similar TensorFlow API 
    (tf.linalg.lu_matrix_inverse) by:
    1. Enabling Mixed Precision (AMP equivalent).
    2. Enforcing Static Shapes via tf.function input_signature.
    3. Using the specific batch size (2) mentioned in the bug report.
    4. Benchmarking performance to detect potential regressions.
    """
    
    # 1. Setup: Mimic AMP (Automatic Mixed Precision)
    # The bug report mentions 'amp'. In TensorFlow, this is handled by the mixed precision policy.
    # We use 'mixed_float16' to simulate the performance-critical environment.
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # 2. Setup: Mimic Static Shapes and Batch Size from Bug Report
    # Bug report data: batch_size_new = 2
    BATCH_SIZE = 2
    # BERT involves large linear layers. We use a representative matrix size for the test.
    MATRIX_SIZE = 512 

    # 3. Data Preparation: Create a batch of invertible matrices
    # Using a fixed seed for reproducibility
    np.random.seed(161878)
    # Generate random matrices and add a multiple of the identity to ensure invertibility
    raw_data = np.random.randn(BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    input_data = raw_data + 10 * np.eye(MATRIX_SIZE, dtype=np.float32)

    # 4. Define the computation graph with Static Shapes
    # Using input_signature enforces static shape compilation, similar to the 'static cpp wrapper'
    # mentioned in the PyTorch issue.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE], dtype=tf.float32)
    ])
    def compute_lu_inverse(x):
        # Perform LU decomposition
        lu, p = tf.linalg.lu(x)
        # Compute inverse using the LU factors (the API under test)
        inv_x = tf.linalg.lu_matrix_inverse(lu, p)
        return inv_x

    # 5. Warmup (Compilation Phase)
    # The bug report highlights 'compilation_latency'. We run once to trigger tracing.
    _ = compute_lu_inverse(input_data)

    # 6. Performance Benchmark
    # The bug report ran 50 iterations. We replicate this to measure speed.
    NUM_ITERATIONS = 50
    start_time = time.time()

    for _ in range(NUM_ITERATIONS):
        result = compute_lu_inverse(input_data)

    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = total_time / NUM_ITERATIONS

    print(f"Test Configuration: Batch Size={BATCH_SIZE}, Matrix Size={MATRIX_SIZE}, AMP=True")
    print(f"Total Time for {NUM_ITERATIONS} iterations: {total_time:.4f}s")
    print(f"Average Latency per iteration: {avg_latency:.6f}s")

    # 7. Correctness Assertion
    # Verify the result against the standard matrix inverse to ensure the API logic holds.
    expected_inv = tf.linalg.inv(tf.cast(input_data, tf.float32))
    # We use a tolerance because mixed precision (float16) can introduce slight numerical differences.
    tf.debugging.assert_near(result, expected_inv, rtol=1e-2, atol=1e-2)

    # 8. Performance Regression Guard
    # The original bug showed a significant slowdown (latency increased from ~0.011s to ~0.030s).
    # We assert that the average latency for this specific operation is reasonable 
    # (e.g., less than 1.0 second for this size on a standard CPU).
    assert avg_latency < 1.0, (
        f"Performance regression detected: Average latency {avg_latency:.6f}s "
        "exceeds acceptable threshold."
    )

if __name__ == "__main__":
    test_lu_matrix_inverse_performance_amp_static()