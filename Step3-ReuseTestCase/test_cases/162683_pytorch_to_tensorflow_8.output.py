import torch
import tensorflow as tf
import time

# Set random seed for reproducibility
tf.random.set_seed(0)

# Define configurations for conv2d_transpose
# Format: (input_shape, kernel_shape, output_shape)
# Adapted from the original shapes to fit conv2d_transpose semantics:
# Original: ((1, 12, 10, 64), (1, 12, 64, 10))
# Adapted: Input (1, 12, 10, 64), Kernel (3, 3, 64, 64), Output (1, 14, 12, 64)
# Note: Output shape calculation for 'valid' padding: (H_in - 1) * stride + kernel_size
configs = [
    ((1, 12, 10, 64), (3, 3, 64, 64), (1, 14, 12, 64)),
    ((1, 12, 10, 10), (3, 3, 10, 10), (1, 14, 12, 10)),
]

def benchmark_conv2d_transpose(input_shape, kernel_shape, output_shape, repeat=500):
    # Create tensors with values in range [-1, 1] similar to torch.uniform_(0,1) * 2 - 1
    x = tf.random.uniform(input_shape, minval=-1, maxval=1)
    kernel = tf.random.uniform(kernel_shape, minval=-1, maxval=1)
    
    # Warm up
    for _ in range(5000):
        _ = tf.keras.backend.conv2d_transpose(x, kernel, output_shape)
    
    # Run benchmark
    times = []
    for i in range(repeat):
        start = time.time()
        _ = tf.keras.backend.conv2d_transpose(x, kernel, output_shape)
        end = time.time()
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
            
    times.sort()
    print(times)
    avg_time_us = sum(times) / len(times)
    return avg_time_us

if __name__ == "__main__":
    # Explicitly run on CPU to match the original bug report context
    with tf.device('/CPU:0'):
        for in_shape, k_shape, out_shape in configs:
            t = benchmark_conv2d_transpose(in_shape, k_shape, out_shape)
            print(f"Input: {in_shape}, Kernel: {k_shape} -> {t:.3f} us")