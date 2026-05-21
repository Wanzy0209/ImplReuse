import torch
import tensorflow as tf
import time
import numpy as np

# Original API Under Test: torch.compile
# Similar API: tf.compat.v1.enable_eager_execution
#
# The original bug report highlights a performance regression in torch.compile 
# for row-wise (dim0) mxfp8 quantization. 
# This test adapts the logic to TensorFlow by enabling eager execution 
# and benchmarking a row-wise quantization operation to verify performance 
# characteristics.

# Enable eager execution as requested by the target API
tf.compat.v1.enable_eager_execution()

def test_dim0_quantization_performance():
    """
    Reproduces the logic of the original bug report: measuring performance 
    of row-wise quantization. 
    Since mxfp8 is specific to PyTorch/torchao, we use TensorFlow's 
    fake_quant_with_min_max_vars_per_channel to simulate the row-wise 
    scaling workload.
    """
    # Parameters from the original bug report
    M = 16384
    K = 16384
    BLOCK_SIZE = 32

    # Create input tensor (simulating the matrix to be quantized)
    # Using float32 to match the precision usually required before quantization
    input_tensor = tf.random.normal((M, K), dtype=tf.float32)

    # Simulate row-wise scaling (dim0)
    # We generate random min/max ranges for each row to simulate the 
    # scaling factors calculated in the original 'dim0_mxfp8_floor' mode.
    min_range = tf.random.uniform((M, 1), minval=-10.0, maxval=0.0)
    max_range = tf.random.uniform((M, 1), minval=0.0, maxval=10.0)

    # Warmup runs to ensure initialization is complete
    for _ in range(10):
        _ = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            input_tensor, min_range, max_range
        )

    # Benchmark loop
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        # Perform the row-wise quantization operation
        output = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            input_tensor, min_range, max_range
        )
    end_time = time.time()

    # Calculate metrics similar to the original cast_bench.py
    avg_time_us = (end_time - start_time) / iterations * 1e6
    
    # Memory bandwidth calculation
    # Input: M*K*4 bytes (float32)
    # Output: M*K*4 bytes (float32)
    total_bytes = (M * K * 4) * 2
    bw_gbps = (total_bytes / 1e9) / ((end_time - start_time) / iterations)

    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"mode: dim0_quantization_tf_eager")
    print(f"time_us {avg_time_us}")
    print(f"mem_bw_gbps {bw_gbps}")

    # Assertions to verify correctness
    assert output.shape == (M, K), f"Output shape mismatch: {output.shape} != {(M, K)}"
    assert output.dtype == tf.float32, f"Output dtype mismatch: {output.dtype}"
    
    # Verify that the operation actually applied quantization (values are clamped)
    # Note: This is a basic sanity check for the operation execution
    assert tf.reduce_all(output >= min_range) and tf.reduce_all(output <= max_range), \
        "Quantization clamping failed."

if __name__ == "__main__":
    test_dim0_quantization_performance()