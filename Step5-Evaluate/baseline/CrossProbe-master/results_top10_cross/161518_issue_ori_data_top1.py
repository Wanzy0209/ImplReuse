```python
import tensorflow as tf
import timeit

# Check for GPU availability to match device="cuda"
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise SystemError("GPU device not found")

# Define dtypes corresponding to PyTorch dtypes
# torch.float8_e4m3fn -> tf.float8_e4m3fn
# torch.bfloat16 -> tf.bfloat16
# torch.float32 -> tf.float32
dtypes = [tf.float8_e4m3fn, tf.bfloat16, tf.float32]

for dtype in dtypes:
    # Conversion: dtype.itemsize -> tf.dtypes.as_dtype(dtype).size
    element_size = tf.dtypes.as_dtype(dtype).size

    with tf.device('/GPU:0'):
        # Conversion: torch.zeros -> tf.zeros
        # Note: Shapes are calculated identically.
        input1 = tf.zeros((128 // element_size) * 1024, 1024, dtype=dtype) # 128 MiB data
        input2 = tf.zeros((128 // element_size) * 1024, 1024, dtype=dtype) # 128 MiB data
        output = tf.zeros((128 // element_size) * 1024, 2048, dtype=dtype) # 256 MiB data

        # Conversion: torch.cat([input1, input2], dim=1, out=output)
        # TensorFlow uses tf.concat. It does not support the 'out' argument to write to an existing tensor.
        # We use tf.function with jit_compile=True to approximate the performance of a captured CUDA graph.
        @tf.function(jit_compile=True)
        def concat_op():
            return tf.concat([input1, input2], axis=1)

        # Warm-up run
        _ = concat_op()

        # Conversion: triton.testing.do_bench_cudagraph(lambda: ..., rep=100)
        # We use timeit to measure execution time. 
        # rep=100 corresponds to number=100 in timeit.
        number = 100
        total_time = timeit.timeit(concat_op, number=number)
        
        # Calculate average time in milliseconds
        output = (total_time / number) * 1000

        bdwidth = 512 / output * 1000 / 1000 / 1000 # TiB/s

        print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))
```