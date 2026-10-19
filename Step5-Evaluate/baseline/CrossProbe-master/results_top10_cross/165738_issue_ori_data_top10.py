```python
import tensorflow as tf
import time

# Conversion: Triton kernel -> TensorFlow function
# The original kernel performs a fused operation: Batch Normalization (inference mode) followed by ReLU.
# Logic: output = relu( gamma * (input - mean) / sqrt(variance + epsilon) + beta )
@tf.function
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel):
    # in_out_ptr0: Input tensor (x)
    # in_ptr0: Mean
    # in_ptr1: Variance
    # in_ptr2: Gamma (scale)
    # in_ptr3: Beta (offset)
    # xnumel: Number of elements (unused in TF, kept for signature compatibility)
    
    epsilon = 0.001
    
    # Perform Batch Normalization
    # tf.nn.batch_normalization arguments: x, mean, variance, offset, scale, variance_epsilon
    # Note: Source uses in_ptr2 as gamma (scale) and in_ptr3 as beta (offset)
    y = tf.nn.batch_normalization(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr3,  # offset (beta)
        in_ptr2,  # scale (gamma)
        epsilon
    )
    
    # Perform ReLU (equivalent to max(0, y))
    return tf.nn.relu(y)

def get_args():
    # Conversion: torch.rand_strided -> tf.random.uniform
    # Shapes match the source: (1024, 32, 149, 149) and (32,)
    # Device: Using '/GPU:0' as standard TF device placement. 
    # Note: 'xpu' in source maps to Intel GPU, standard TF uses '/GPU:0' or '/device:GPU:0'.
    arg_0 = tf.random.uniform((1024, 32, 149, 149), dtype=tf.float16)
    arg_1 = tf.random.uniform((32,), dtype=tf.float16)
    arg_2 = tf.random.uniform((32,), dtype=tf.float16) + 1.0
    arg_3 = tf.random.uniform((32,), dtype=tf.float16)
    arg_4 = tf.random.uniform((32,), dtype=tf.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368

def call(args):
    # Conversion: torch.xpu._DeviceGuard -> tf.device context manager
    # We use '/GPU:0' to represent the primary accelerator device.
    with tf.device('/GPU:0'):
        # Unpack arguments
        # The source passes xnumel as the last argument
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)

def benchmark_all_configs(args):
    # Conversion: PyTorch Inductor autotuning -> TF execution
    # TF handles kernel selection and optimization (XLA) internally.
    with tf.device('/GPU:0'):
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)

if __name__ == '__main__':
    # Conversion: Benchmarking logic
    args = get_args()
    
    # Warm-up runs to allow XLA compilation (if enabled) and GPU initialization
    for _ in range(10):
        _ = call(args)
        
    # Benchmarking loop
    # We measure the time taken for 40 repetitions
    start_time = time.perf_counter()
    for _ in range(40):
        _ = call(args)
    end_time = time.perf_counter()
    
    # Calculate metrics
    ms = (end_time - start_time) * 1000  # Total time in milliseconds
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1000)
    
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```