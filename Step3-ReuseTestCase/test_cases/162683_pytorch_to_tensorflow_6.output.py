import torch
import tensorflow as tf
import time

# Set seed for reproducibility
tf.random.set_seed(0)

# Adapted shapes from the original bug report.
# Since resize_images takes a single 4D tensor, we use the first shape from each pair.
shapes = [
    (1, 12, 10, 64),
    (1, 12, 10, 10),
]

def benchmark_resize_images(input_shape, dtype=tf.float16, repeat=500):
    """
    Benchmarks the tf.keras.backend.resize_images operation.
    Adapted from the torch.matmul benchmark to test the similar API.
    """
    # Ensure operation runs on CPU to match the original bug context
    with tf.device("/cpu:0"):
        # Create input tensor
        X = tf.random.uniform(input_shape, minval=-1, maxval=1, dtype=dtype)
        
        # Define resize factors
        height_factor = 2
        width_factor = 2
        data_format = 'channels_first' # Assuming PyTorch default (NCHW) semantics for 4D tensors

        # Verify behavior: Check if output shape is correct
        # Expected: (Batch, Channels, Height * factor, Width * factor)
        expected_shape = list(input_shape)
        if data_format == 'channels_first':
            expected_shape[2] *= height_factor
            expected_shape[3] *= width_factor
        else:
            expected_shape[1] *= height_factor
            expected_shape[2] *= width_factor
            
        # Warm up
        for _ in range(5000):
            _ = tf.keras.backend.resize_images(X, height_factor, width_factor, 
                                               data_format=data_format, interpolation='bilinear')

        # Run benchmark
        times = []
        for i in range(repeat):
            start = time.time()
            output = tf.keras.backend.resize_images(X, height_factor, width_factor, 
                                                    data_format=data_format, interpolation='bilinear')
            end = time.time()
            
            # Verify output shape once during the loop
            if i == 0:
                assert list(output.shape) == expected_shape, \
                    f"Shape mismatch: expected {expected_shape}, got {list(output.shape)}"

            if i > 100:
                times.append(round((end - start) * 1000 * 1000)) # microseconds

    times.sort()
    print(f"Times (us) for shape {input_shape}: {times[:10]}...") # Print first 10 for brevity
    avg_time_us = sum(times) / len(times)
    return avg_time_us

if __name__ == "__main__":
    for shape in shapes:
        t = benchmark_resize_images(shape)
        print(f"Resize {shape} -> {t:.3f} us")