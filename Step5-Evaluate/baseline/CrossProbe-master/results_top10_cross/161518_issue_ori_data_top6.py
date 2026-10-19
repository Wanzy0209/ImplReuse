```python
import tensorflow as tf
import time

for dtype in [tf.float8_e4m3fn, tf.bfloat16, tf.float32]:
    # Conversion: dtype.itemsize -> tf.dtypes.as_dtype(dtype).size
    element_size = tf.dtypes.as_dtype(dtype).size

    # Conversion: device="cuda" -> tf.device('/GPU:0')
    with tf.device('/GPU:0'):
        # Conversion: torch.zeros -> tf.zeros
        input1 = tf.zeros(((128 // element_size) * 1024, 1024), dtype=dtype) # 128 MiB data
        input2 = tf.zeros(((128 // element_size) * 1024, 1024), dtype=dtype) # 128 MiB data
        output = tf.zeros(((128 // element_size) * 1024, 2048), dtype=dtype) # 256 MiB data

        # Conversion: torch.cat([input1, input2], dim=1, out=output)
        # Note: tf.concat does not support an 'out' argument for in-place writing.
        # It returns a new tensor. We wrap in tf.function to simulate graph execution.
        @tf.function
        def concat_op():
            return tf.concat([input1, input2], axis=1) # 512 MiB IO

        # Warmup run to trace the graph
        concat_op()

        # Conversion: triton.testing.do_bench_cudagraph
        # We implement a manual timing loop since there is no direct TF equivalent.
        # rep=100
        start = time.perf_counter()
        for _ in range(100):
            concat_op()
        end = time.perf_counter()

        # Reusing 'output' variable to store the average time in ms, matching source structure
        output = (end - start) / 100 * 1000

        bdwidth = 512 / output * 1000 / 1000 / 1000 # TiB/s

        print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))
```