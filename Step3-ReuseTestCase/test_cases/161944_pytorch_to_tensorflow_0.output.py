import torch
import tensorflow as tf
import numpy as np

# Adapted test case for tf.raw_ops.HashTable based on the torch.exp precision issue.
# The original test compares eager execution, compiled execution, and high-precision execution.
# Here we verify the behavior of tf.raw_ops.HashTable across these modes.

def run_test():
    # Setup inputs
    # Using random floats to mimic the original torch.randn input
    np.random.seed(42)
    num_elements = 8192
    
    # Float32 inputs
    keys_32 = np.random.rand(num_elements).astype(np.float32)
    values_32 = np.random.rand(num_elements).astype(np.float32)
    query_keys_32 = keys_32[:100] # Lookup a subset of keys

    # Float64 inputs (High precision)
    keys_64 = keys_32.astype(np.float64)
    values_64 = values_32.astype(np.float64)
    query_keys_64 = query_keys_32.astype(np.float64)

    # Define the operation function using tf.raw_ops.HashTable
    def hash_op(keys, values, query, dtype):
        # Create a hash table. 
        # empty_key is set to -1.0 assuming it's not in the random data.
        table = tf.raw_ops.HashTable(
            key_dtype=dtype,
            value_dtype=dtype,
            empty_key=-1.0
        )
        
        init = table.init
        insert = table.insert(keys, values)
        lookup = table.lookup(query)
        
        # Ensure initialization and insertion happen before lookup
        with tf.control_dependencies([init, insert]):
            return tf.identity(lookup)

    # 1. Eager execution (Float32)
    out1 = hash_op(keys_32, values_32, query_keys_32, tf.float32)

    # 2. Compiled execution (Float32)
    # In TensorFlow, tf.function compiles the graph, analogous to torch.compile
    @tf.function
    def compiled_hash_op():
        return hash_op(keys_32, values_32, query_keys_32, tf.float32)

    out2 = compiled_hash_op()

    # 3. High precision execution (Float64)
    out3_high = hash_op(keys_64, values_64, query_keys_64, tf.float64)

    # Compare results
    # We cast the float32 results to float64 to compare with the high precision ground truth
    diff_eager = tf.reduce_max(tf.abs(out3_high - tf.cast(out1, tf.float64)))
    diff_compiled = tf.reduce_max(tf.abs(out3_high - tf.cast(out2, tf.float64)))

    print(f"Max difference (High Precision - Eager FP32): {diff_eager.numpy()}")
    print(f"Max difference (High Precision - Compiled FP32): {diff_compiled.numpy()}")

    # Assertions to verify correctness
    # HashTable lookups should be exact, so differences should be negligible (near machine epsilon)
    assert diff_eager.numpy() < 1e-6, "Eager FP32 precision differs significantly from FP64"
    assert diff_compiled.numpy() < 1e-6, "Compiled FP32 precision differs significantly from FP64"

if __name__ == "__main__":
    run_test()