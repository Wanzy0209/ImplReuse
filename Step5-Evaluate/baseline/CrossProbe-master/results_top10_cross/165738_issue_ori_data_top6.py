```python
import tensorflow as tf
import time

# Conversion comment: Replacing Triton kernel with TensorFlow operations
# The original kernel performs: Batch Normalization (inference) + ReLU
# Formula: output = relu((input - mean) / sqrt(variance + epsilon) * weight + bias)
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel):
    # in_out_ptr0: input tensor
    # in_ptr0: mean
    # in_ptr1: variance
    # in_ptr2: weight (gamma)
    # in_ptr3: bias (beta)
    # xnumel: number of elements (unused in TF high-level ops, derived from shape)

    epsilon = 0.001

    # TensorFlow's batch_normalization expects:
    # x, mean, variance, offset, scale, variance_epsilon
    # Mapping source args to TF args:
    # scale = in_ptr2 (weight)
    # offset = in_ptr3 (bias)
    
    # The source data layout based on strides (710432, 1, 4768, 32) for shape (1024, 32, 149, 149)
    # corresponds to a custom layout, but standard TF usage implies NCHW or NHWC.
    # We assume NCHW here as the shape is (N, C, H, W).
    result = tf.nn.batch_normalization(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        offset=in_ptr3,
        scale=in_ptr2,
        variance_epsilon=epsilon,
        data_format='NCHW'
    )

    # Apply ReLU (equivalent to max(0, x))
    result = tf.nn.relu(result)
    return result


def get_args():
    # Conversion comment: Generating random tensors using TensorFlow
    # Shape (1024, 32, 149, 149) matches source
    # Using standard NCHW layout for TensorFlow
    arg_0 = tf.random.uniform((1024, 32, 149, 149), dtype=tf.float16)
    arg_1 = tf.random.uniform((32,), dtype=tf.float16) # mean
    arg_2 = tf.random.uniform((32,), dtype=tf.float16) + 1.0 # variance
    arg_3 = tf.random.uniform((32,), dtype=tf.float16) # weight
    arg_4 = tf.random.uniform((32,), dtype=tf.float16) # bias
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368


def call(args):
    # Conversion comment: Removing explicit device guard and stream management
    # TensorFlow manages device placement and streams automatically
    with tf.device('/GPU:0'):
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)


def benchmark_all_configs(args):
    # Conversion comment: TensorFlow runtime handles optimization/autotuning internally
    with tf.device('/GPU:0'):
        return call(args)


if __name__ == '__main__':
    # Conversion comment: Replacing custom benchmarker with standard Python timing
    args = get_args()
    
    # Warm-up run to allow XLA/TF compilation
    call(args)

    # Benchmarking loop
    start_time = time.time()
    for _ in range(40):
        res = call(args)
    # Force execution to complete before stopping timer
    _ = res.numpy()
    end_time = time.time()

    ms = (end_time - start_time) * 1000 / 40
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```