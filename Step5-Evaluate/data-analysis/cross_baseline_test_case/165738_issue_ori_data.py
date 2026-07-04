```python
import tensorflow as tf
import time

# Conversion: Replaced Triton imports and decorators with TensorFlow equivalents
# The original Triton kernel implements a fused Batch Normalization (inference) + ReLU.
# We use tf.nn.batch_normalization and tf.nn.relu.

# Conversion: @triton_heuristics.pointwise and @triton.jit
# We use @tf.function to compile the graph, similar to a JIT compiled kernel.
@tf.function(experimental_compile=True)
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel):
    # xnumel is passed for signature compatibility but not used as TF infers shape.
    
    # Conversion: Cast inputs to float32 to match source logic: .to(tl.float32)
    x = tf.cast(in_out_ptr0, tf.float32)
    mean = tf.cast(in_ptr0, tf.float32)
    var = tf.cast(in_ptr1, tf.float32)
    gamma = tf.cast(in_ptr2, tf.float32)
    beta = tf.cast(in_ptr3, tf.float32)

    # Conversion: tmp7 = 0.001
    epsilon = 0.001

    # Conversion: Batch Normalization logic
    # Source: (x - mean) / sqrt(var + eps) * gamma + beta
    # TF: batch_normalization(x, mean, variance, offset, scale, variance_epsilon)
    # Note: offset=beta, scale=gamma
    y = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

    # Conversion: ReLU logic (triton_helpers.maximum(0, y))
    out = tf.nn.relu(y)

    # Conversion: Cast back to fp16 to match source store type
    return tf.cast(out, tf.float16)


def get_args():
    # Conversion: torch._dynamo.testing.rand_strided -> tf.random.uniform
    # Shape: (1024, 32, 149, 149), dtype: float16
    arg_0 = tf.random.uniform((1024, 32, 149, 149), minval=-1.0, maxval=1.0, dtype=tf.float16)
    arg_1 = tf.random.uniform((32,), minval=-1.0, maxval=1.0, dtype=tf.float16)
    arg_2 = tf.random.uniform((32,), minval=-1.0, maxval=1.0, dtype=tf.float16) + 1.0
    arg_3 = tf.random.uniform((32,), minval=-1.0, maxval=1.0, dtype=tf.float16)
    arg_4 = tf.random.uniform((32,), minval=-1.0, maxval=1.0, dtype=tf.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368,


def call(args):
    # Conversion: torch.xpu.set_device and stream handling -> tf.device
    with tf.device('/GPU:0'):
        # Run the compiled kernel
        # Note: TF tensors are immutable, so we capture the result to ensure execution
        _ = triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)


def benchmark_all_configs(args):
    # Conversion: Benchmarking wrapper
    with tf.device('/GPU:0'):
        # In TF, we don't typically benchmark "all configs" for a single op like Inductor does for Triton.
        # We simply run the op.
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)


if __name__ == '__main__':
    # Conversion: Benchmarking logic
    args = get_args()
    
    # Warmup
    call(args)

    # Benchmark
    rep = 40
    start_time = time.time()
    for _ in range(rep):
        call(args)
    end_time = time.time()
    
    ms = (end_time - start_time) * 1000 / rep
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```