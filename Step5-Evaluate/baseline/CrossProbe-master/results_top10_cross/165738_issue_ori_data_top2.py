```python
import tensorflow as tf
import time

# Conversion: Triton kernel logic translated to TensorFlow ops.
# The original kernel performs: Batch Normalization (inference) + ReLU.
# Formula: output = ReLU(gamma * (x - mean) / sqrt(var + epsilon) + beta)
@tf.function
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel):
    # Cast inputs to float32 for computation stability, matching source .to(tl.float32)
    x = tf.cast(in_out_ptr0, tf.float32)
    mean = tf.cast(in_ptr0, tf.float32)
    var = tf.cast(in_ptr1, tf.float32)
    gamma = tf.cast(in_ptr2, tf.float32)
    beta = tf.cast(in_ptr3, tf.float32)

    epsilon = 0.001

    # Batch Normalization
    # tf.nn.batch_normalization computes: scale * (x - mean) / sqrt(variance + epsilon) + offset
    # Mapping: scale=gamma, offset=beta
    y = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

    # ReLU activation
    # Equivalent to triton_helpers.maximum(0, y)
    y = tf.nn.relu(y)

    # Cast back to float16
    return tf.cast(y, tf.float16)


def get_args():
    # Conversion: torch.rand_strided with specific strides replaced by standard TF tensor creation.
    # TF manages memory layout automatically.
    with tf.device('/GPU:0'):
        arg_0 = tf.random.uniform((1024, 32, 149, 149), dtype=tf.float16)
        arg_1 = tf.random.uniform((32,), dtype=tf.float16)
        arg_2 = tf.random.uniform((32,), dtype=tf.float16) + 1.0
        arg_3 = tf.random.uniform((32,), dtype=tf.float16)
        arg_4 = tf.random.uniform((32,), dtype=tf.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368


def call(args):
    # Conversion: torch.xpu._DeviceGuard and set_device replaced by tf.device context.
    # Stream handling is abstracted by TensorFlow runtime.
    with tf.device('/GPU:0'):
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)


def benchmark_all_configs(args):
    # Conversion: Triton autotuning/benchmarking replaced by standard execution.
    # TensorFlow optimizes graph execution (XLA) automatically.
    with tf.device('/GPU:0'):
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)


if __name__ == '__main__':
    # Conversion: torch._inductor.runtime.benchmarking replaced by manual timing.
    args = get_args()

    # Warm-up run
    call(args)

    # Benchmarking
    start_time = time.time()
    for _ in range(40):
        _ = call(args)
    end_time = time.time()

    ms = (end_time - start_time) * 1000 / 40
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```