import tensorflow as tf
import time

# Set seed for reproducibility
tf.random.set_seed(0)

# Adapted configurations for tf.nn.conv2d_transpose
# Original shapes were pairs for Matmul: (A, B).
# For Conv2D Transpose, we map A to Input, and derive Filter/Output shapes based on B's dimensions.
# Original Pair 1: (1, 12, 10, 64) x (1, 12, 64, 10)
#   -> Input: (1, 12, 10, 64), Output Channels: 10
#   -> Filter: (3, 3, 10, 64), Output Shape: (1, 14, 12, 10) [Valid padding, stride 1]
# Original Pair 2: (1, 12, 10, 10) x (1, 12, 10, 64)
#   -> Input: (1, 12, 10, 10), Output Channels: 64
#   -> Filter: (3, 3, 64, 10), Output Shape: (1, 14, 12, 64) [Valid padding, stride 1]
configs = [
    ((1, 12, 10, 64), (3, 3, 10, 64), (1, 14, 12, 10)),
    ((1, 12, 10, 10), (3, 3, 64, 10), (1, 14, 12, 64)),
]

def benchmark_conv2d_transpose(input_shape, filter_shape, output_shape, dtype=tf.float16, repeat=500):
    # Create tensors
    input_tensor = tf.random.uniform(input_shape, minval=-1, maxval=1, dtype=dtype)
    kernel = tf.random.uniform(filter_shape, minval=-1, maxval=1, dtype=dtype)
    
    # Define strides and padding
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    # Warm up
    for _ in range(5000):
        _ = tf.nn.conv2d_transpose(input_tensor, kernel, output_shape=output_shape, strides=strides, padding=padding)

    # Run benchmark
    times = []
    for i in range(repeat):
        start = time.time()
        _ = tf.nn.conv2d_transpose(input_tensor, kernel, output_shape=output_shape, strides=strides, padding=padding)
        end = time.time()
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
            
    times.sort()
    print(times)
    avg_time_us = sum(times) / len(times)
    return avg_time_us

if __name__ == "__main__":
    for input_shape, filter_shape, output_shape in configs:
        t = benchmark_conv2d_transpose(input_shape, filter_shape, output_shape)
        print(f"Input: {input_shape}, Filter: {filter_shape} -> {t:.3f} us")