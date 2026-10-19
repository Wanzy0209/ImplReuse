```python
import tensorflow as tf
import time

# Conversion: The Triton kernel implements a fused Batch Normalization (inference mode) + ReLU.
# In TensorFlow, this is best represented using tf.nn.batch_normalization followed by tf.nn.relu.
# The original kernel operates on float16 data.

@tf.function
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3):
    """
    TensorFlow equivalent of the Triton kernel.
    Args:
        in_out_ptr0: Input tensor [N, C, H, W]
        in_ptr0: Mean [C]
        in_ptr1: Variance [C]
        in_ptr2: Gamma (scale) [C]
        in_ptr3: Beta (offset) [C]
    """
    # Epsilon from source: tmp7 = 0.001
    epsilon = 0.001

    # Batch Normalization: ((x - mean) / sqrt(var + epsilon)) * gamma + beta
    # data_format='NCHW' matches the input shape (1024, 32, 149, 149)
    x_bn = tf.nn.batch_normalization(
        in_out_ptr0,
        mean=in_ptr0,
        variance=in_ptr1,
        offset=in_ptr3,
        scale=in_ptr2,
        variance_epsilon=epsilon,
        data_format='NCHW'
    )

    # ReLU: max(0, x)
    # Source: tmp23 = triton_helpers.maximum(0, tmp21)
    return tf.nn.relu(x_bn)

def get_args():
    # Conversion: rand_strided creates tensors on specific devices.
    # We use tf.device to place them on the GPU.
    # Shape: (1024, 32, 149, 149) for input, (32,) for params.
    with tf.device('/GPU:0'):
        arg_0 = tf.random.uniform((1024, 32, 149, 149), dtype=tf.float16)
        arg_1 = tf.random.uniform((32,), dtype=tf.float16) # mean
        arg_2 = tf.random.uniform((32,), dtype=tf.float16) + 1.0 # var
        arg_3 = tf.random.uniform((32,), dtype=tf.float16) # gamma
        arg_4 = tf.random.uniform((32,), dtype=tf.float16) # beta
    return arg_0, arg_1, arg_2, arg_3, arg_4

def call(args):
    # Conversion: The original code uses specific stream and device guards.
    # In TF, graph execution (via @tf.function) handles device placement and execution.
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3 = args
    return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3)

def benchmark_all_configs(args):
    # Conversion: TF does not expose kernel-level config benchmarking like Triton.
    # We simply run the op.
    return call(args)

if __name__ == '__main__':
    # Conversion: Benchmarking logic adapted for Python/TensorFlow.
    args = get_args()

    # Warmup
    call(args)

    # Benchmark
    start_time = time.time()
    rep = 40
    for _ in range(rep):
        _ = call(args)
    end_time = time.time()

    ms = (end_time - start_time) * 1000 / rep
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```