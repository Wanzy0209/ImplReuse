import torch
import tensorflow as tf
import time

def test_categorical_memory_throughput():
    """
    Test case adapted from PyTorch Issue 161518 (Memory Throughput on B200 for copy_).
    
    This test benchmarks the memory throughput of operations involving 
    tf.compat.v1.distributions.Categorical, mirroring the logic used to 
    benchmark torch.cat in the original issue.
    """
    
    # We test with bfloat16 and float32 to match the issue's intent of checking 
    # performance across different element sizes.
    # Note: tf.float8_e4m3fn is available in newer TF versions but omitted here 
    # to ensure broader compatibility of the test case.
    dtypes = [tf.bfloat16, tf.float32]

    for dtype in dtypes:
        element_size = dtype.size

        # Create large tensors (128 MiB data each)
        # Shape calculation: (128 * 1024 * 1024) bytes / element_size
        # We use a shape that allows concatenation along dim 1, similar to the issue.
        dim0 = (128 * 1024 * 1024) // (element_size * 1024)
        input1 = tf.zeros((dim0, 1024), dtype=dtype)
        input2 = tf.zeros((dim0, 1024), dtype=dtype)

        # Concatenate to mimic the torch.cat operation in the issue
        # This results in 256 MiB of logits
        combined_logits = tf.concat([input1, input2], axis=1)

        # Instantiate the distribution using the similar API
        # This acts as the "operation" under test, analogous to the copy/cat ops.
        dist = tf.compat.v1.distributions.Categorical(logits=combined_logits)

        # Warmup run to ensure initialization is complete
        _ = dist.sample()

        # Benchmark loop
        # We measure sampling, which reads the logits (256 MiB) and writes samples.
        # Assuming int32 samples (4 bytes), output is roughly 256 MiB.
        # Total IO ~ 512 MiB, matching the issue's IO calculation.
        repetitions = 100
        start_time = time.time()
        for _ in range(repetitions):
            _ = dist.sample()
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000 / repetitions
        
        # Calculate bandwidth in TiB/s to match the issue's reporting
        # 512 MiB = 512 / 1024 / 1024 TiB
        io_tib = 512 / 1024 / 1024
        bdwidth = io_tib / duration_ms * 1000

        print("\t".join([str(dtype), f"{duration_ms:.2f} ms", f"{bdwidth:.4f} TiB/s"]))
        
        # Basic assertion to ensure the operation executed and produced valid output
        # (Checking that time passed and operation didn't crash)
        assert duration_ms > 0, "Benchmark duration should be positive"

if __name__ == "__main__":
    test_categorical_memory_throughput()