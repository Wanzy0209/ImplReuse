```python
import tensorflow as tf
import time

# Conversion: Triton kernel logic -> TensorFlow function
# The original kernel implements a fused Batch Normalization (inference mode) followed by ReLU.
# Logic: output = ReLU( ((input - mean) / sqrt(variance + epsilon)) * weight + bias )
@tf.function
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel):
    # xnumel is kept for signature compatibility but is not used in TF logic (shape is inferred)
    
    # TensorFlow's batch_normalization performs: scale * (x - mean) / sqrt(variance + epsilon) + offset
    # Mapping source arguments to TF arguments:
    # in_out_ptr0 (input) -> x
    # in_ptr0 (mean)      -> mean
    # in_ptr1 (variance)  -> variance
    # in_ptr2 (weight)    -> scale
    # in_ptr3 (bias)      -> offset
    # epsilon             -> 0.001 (from source tmp7)
    
    bn_output = tf.nn.batch_normalization(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        offset=in_ptr3,
        scale=in_ptr2,
        variance_epsilon=0.001
    )
    
    # Conversion: triton_helpers.maximum(tmp22, tmp21) -> tf.nn.relu
    # tmp22 is 0, tmp21 is the BN output
    output = tf.nn.relu(bn_output)
    
    return output

def get_args():
    # Conversion: torch.rand_strided -> tf.random.uniform
    # Conversion: device='xpu:0' -> tf.device('/device:GPU:0')
    # Note: Strides are handled implicitly by TensorFlow's memory layout.
    with tf.device('/device:GPU:0'):
        arg_0 = tf.random.uniform((1024, 32, 149, 149), dtype=tf.float16)
        arg_1 = tf.random.uniform((32,), dtype=tf.float16)
        arg_2 = tf.random.uniform((32,), dtype=tf.float16) + 1.0
        arg_3 = tf.random.uniform((32,), dtype=tf.float16)
        arg_4 = tf.random.uniform((32,), dtype=tf.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368

def call(args):
    # Conversion: torch.xpu._DeviceGuard and set_device -> tf.device context manager
    # Conversion: get_raw_stream -> Implicit in TF execution
    with tf.device('/device:GPU:0'):
        # Execute the op. TF returns the result tensor (in-place modification is not standard in TF)
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0(*args)

def benchmark_all_configs(args):
    # Conversion: PyTorch kernel benchmarking -> Direct execution
    with tf.device('/device:GPU:0'):
        return call(args)

if __name__ == '__main__':
    # Conversion: torch._inductor.runtime.benchmarking -> Manual timing loop
    args = get_args()
    
    # Warmup runs
    for _ in range(10):
        call(args)

    # Benchmarking
    start_time = time.time()
    for _ in range(40):
        call(args)
    end_time = time.time()
    
    ms = (end_time - start_time) / 40 * 1000
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
```