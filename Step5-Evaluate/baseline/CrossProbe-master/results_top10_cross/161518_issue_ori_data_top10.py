```python
import tensorflow as tf
import time

# Helper function to mimic triton.testing.do_bench_cudagraph
def do_bench(fn, rep=100):
    # Warmup
    for _ in range(10):
        fn()
    
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000 / rep

for dtype in [tf.float8_e4m3fn, tf.bfloat16, tf.float32]:
    # Conversion: PyTorch dtype.itemsize -> TensorFlow dtype size lookup
    element_size = tf.dtypes.as_dtype(dtype).size

    input1 = tf.zeros((128 // element_size) * 1024, 1024, dtype=dtype) # 128 MiB data
    input2 = tf.zeros((128 // element_size) * 1024, 1024, dtype=dtype) # 128 MiB data
    # Conversion: Pre-allocation of output is not supported in tf.concat.
    # We define the tensor here to match the original structure, but it will be overwritten by the result.
    output = tf.zeros((128 // element_size) * 1024, 2048, dtype=dtype) # 256 MiB data

    # Conversion: Wrap in tf.function for optimized execution (similar to CUDA Graph capture)
    @tf.function(jit_compile=True)
    def concat_op():
        # Conversion: torch.cat with out argument -> tf.concat (returns new tensor)
        return tf.concat([input1, input2], axis=1)

    output = do_bench(lambda: concat_op(), rep=100)

    bdwidth = 512 / output * 1000 / 1000 / 1000 # TiB/s

    print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))
```