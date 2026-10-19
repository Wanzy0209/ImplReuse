```python
import tensorflow as tf
import time

# Conversion: Replaced triton and torch._inductor imports with tensorflow
# Conversion: Removed triton_heuristics and DeviceProperties as they are PyTorch/Triton specific

# Conversion: Replaced Triton kernel with TensorFlow operations
# The original kernel performs: Output = ReLU(BatchNorm(Input, Mean, Var, Gamma, Beta))
# Note: TensorFlow's batch_normalization normalizes over the last dimension by default.
# To match the logic, we ensure the channel dimension (size 32) is the last dimension (NHWC format).
@tf.function
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel):
    # Cast inputs to float32 for calculation (matching source .to(tl.float32))
    x = tf.cast(in_out_ptr0, tf.float32)
    mean = tf.cast(in_ptr0, tf.float32)
    variance = tf.cast(in_ptr1, tf.float32)
    gamma = tf.cast(in_ptr2, tf.float32)
    beta = tf.cast(in_ptr3, tf.float32)
    
    # Epsilon from source code (tmp7 = 0.001)
    epsilon = 0.001
    
    # Batch Normalization: (x - mean) / sqrt(variance + epsilon) * gamma + beta
    # tf.nn.batch_normalization arguments: x, mean, variance, offset (beta), scale (gamma), variance_epsilon
    bn_out = tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=epsilon)
    
    # ReLU activation (matching triton_helpers.maximum)
    relu_out = tf.nn.relu(bn_out)
    
    # Cast back to fp16
    return tf.cast(relu_out, tf.float16)


def get_args():
    # Conversion: Replaced torch.rand_strided with tf.random.uniform
    # Original shape (1024, 32, 149, 149) in NCHW format.
    # Converted to (1024, 149, 149, 32) NHWC for TensorFlow default layout.
    arg_0 = tf.random.uniform((1024, 149, 149, 32), dtype=tf.float16)
    
    # Mean, Variance, Gamma, Beta with shape (32,)
    arg_1 = tf.random.uniform((32,), dtype=tf.float16)
    arg_2 = tf.random.uniform((32,), dtype=tf.float16) + 1.0
    arg_3 = tf.random.uniform((32,), dtype=tf.float16)
    arg_4 = tf.random.uniform((32,), dtype=tf.float16)
    
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368


def call(args):
    # Conversion: Replaced torch.xpu._DeviceGuard and set_device with tf.device
    with tf.device('/GPU:0'):
        x, mean, var, scale, offset, numel = args
        # Execute the function
        _ = triton_poi_fused__native_batch_norm_legit_no_training_relu_0(
            x, mean, var, scale, offset, numel
        )


def benchmark_all_configs(args):
    # Conversion: Replaced kernel.benchmark_all_configs with a simple execution
    # TensorFlow does not have a direct 1:1 for Triton's config benchmarking in eager mode
    with tf.device('/GPU:0'):
        x, mean, var, scale, offset, numel = args
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(
            x, mean, var, scale, offset, numel
        )


if __name__ == '__main__':
    # Conversion: Replaced torch._inductor.runtime.benchmarking with manual timing loop
    args = get_args()
    
    # Warm-up
    for _ in range(5):
        call(args)

    # Benchmark
    start_time = time.perf_counter()
    rep = 40
    for _ in range(rep):
        call(args)
    end_time = time.perf_counter()
    
    ms = (end_time - start_time) / rep * 1000.0
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```