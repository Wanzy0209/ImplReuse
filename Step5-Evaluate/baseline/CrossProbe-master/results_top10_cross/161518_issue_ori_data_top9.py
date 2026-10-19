```python
import tensorflow as tf
import time

# Note: tf.float8_e4m3fn is available in TensorFlow 2.16+
for dtype in [tf.float8_e4m3fn, tf.bfloat16, tf.float32]:
    element_size = dtype.size

    # Ensure operations are placed on the GPU
    with tf.device('/GPU:0'):
        input1 = tf.zeros([(128 // element_size) * 1024, 1024], dtype=dtype) # 128 MiB data
        input2 = tf.zeros([(128 // element_size) * 1024, 1024], dtype=dtype) # 128 MiB data
        output = tf.zeros([(128 // element_size) * 1024, 2048], dtype=dtype) # 256 MiB data

        # TensorFlow does not support in-place concatenation via an 'out' argument.
        # We use tf.concat inside a tf.function to simulate graph execution (similar to CUDAGraphs).
        @tf.function
        def concat_op():
            return tf.concat([input1, input2], axis=1)

        # Benchmarking logic to replace triton.testing.do_bench_cudagraph
        # Warmup
        for _ in range(10):
            concat_op()

        # Timing
        start = time.time()
        for _ in range(100):
            concat_op()
        end = time.time()

        output = (end - start) / 100 * 1000 # Average time in ms

        bdwidth = 512 / output * 1000 / 1000 / 1000 # TiB/s

        print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))
```