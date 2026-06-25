```python
import tensorflow as tf
import time

# Conversion: Iterate over TensorFlow equivalents of PyTorch dtypes
# Note: tf.float8_e4m3fn is available in TensorFlow 2.16+
for dtype in [tf.float8_e4m3fn, tf.bfloat16, tf.float32]:
    # Conversion: PyTorch dtype.itemsize -> TF dtype.size
    # tf.float8_e4m3fn size is 1 byte
    if dtype == tf.float8_e4m3fn:
        element_size = 1
    else:
        element_size = tf.dtypes.as_dtype(dtype).size

    # Conversion: torch.zeros -> tf.zeros
    # Conversion: device="cuda" -> tf.device('/GPU:0')
    with tf.device('/GPU:0'):
        input1 = tf.zeros(((128 // element_size) * 1024, 1024), dtype=dtype) # 128 MiB data
        input2 = tf.zeros(((128 // element_size) * 1024, 1024), dtype=dtype) # 128 MiB data
        
        # Conversion: torch.zeros with out argument -> tf.Variable
        # TF ops generally return new tensors. To write to a pre-allocated buffer, we use a Variable.
        output = tf.Variable(tf.zeros(((128 // element_size) * 1024, 2048), dtype=dtype)) # 256 MiB data

        # Conversion: torch.cat -> tf.concat
        # Conversion: out=output -> output.assign(...)
        # Conversion: triton.testing.do_bench_cudagraph -> tf.function + time measurement
        # We use tf.function with jit_compile=True to mimic the graph execution/capture
        @tf.function(jit_compile=True)
        def concat_op():
            output.assign(tf.concat([input1, input2], axis=1))

        # Warmup run
        concat_op()

        # Benchmarking loop (replacing triton.testing.do_bench_cudagraph)
        start_time = time.time()
        for _ in range(100):
            concat_op()
        end_time = time.time()
        
        # Calculate average time in ms
        elapsed_ms = (end_time - start_time) * 1000 / 100

        # 512 MiB IO (128+128 read, 256 write)
        bdwidth = 512 / elapsed_ms * 1000 / 1000 / 1000 # TiB/s

        print("\t".join([str(dtype), f"{elapsed_ms:.2f} ms", f"{bdwidth:.2f} TiB/s"]))
```