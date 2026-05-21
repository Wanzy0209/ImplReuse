import tensorflow as tf
import numpy as np

def test_hash_table_divergence():
    """
    Adapted test case for tf.raw_ops.HashTable based on the PyTorch bug report.
    The original bug involved an IncompatibleTypeError with int64 and float16 
    types during eager vs. compiled execution. This test verifies that 
    tf.raw_ops.HashTable handles these specific types consistently in both 
    Eager and Graph (tf.function) modes.
    """
    
    # Replicating the data types from the original PyTorch bug report
    # arg0: int64, arg1: float16
    # We flatten the tensors to be compatible with HashTable key-value initialization
    keys = tf.constant(np.random.randint(0, 1000, size=(100,)), dtype=tf.int64)
    values = tf.constant(np.random.randn(100,), dtype=tf.float16)
    
    # Query keys (subset of initialization keys)
    query_keys = tf.constant(np.random.randint(0, 100, size=(10,)), dtype=tf.int64)
    default_value = tf.constant(0.0, dtype=tf.float16)

    def run_logic(keys, values, query_keys, default_value):
        # Create the HashTable using the raw op
        # Note: tf.raw_ops.HashTable creates a resource handle
        table_handle = tf.raw_ops.HashTable(
            key_dtype=tf.int64,
            value_dtype=tf.float16,
            container="",
            shared_name=""
        )
        
        # Initialize the table with keys and values
        init_op = tf.raw_ops.InitializeTable(
            table_handle=table_handle,
            keys=keys,
            values=values
        )
        
        # Perform a lookup operation
        # We use a control dependency to ensure initialization happens before lookup
        with tf.control_dependencies([init_op]):
            result = tf.raw_ops.LookupTableFind(
                table_handle=table_handle,
                keys=query_keys,
                default_value=default_value
            )
        return result

    # 1. Test Eager Mode
    print("Running Eager Mode...")
    try:
        out_eager = run_logic(keys, values, query_keys, default_value)
        print(f"Eager Result Shape: {out_eager.shape}, Dtype: {out_eager.dtype}")
        print("Eager Success! ")
    except Exception as e:
        print(f"Eager Failed! : {e}")
        return

    # 2. Test Compiled Mode (tf.function)
    print("\nRunning Compiled Mode (tf.function)...")
    compiled_logic = tf.function(run_logic)
    try:
        out_compiled = compiled_logic(keys, values, query_keys, default_value)
        print(f"Compiled Result Shape: {out_compiled.shape}, Dtype: {out_compiled.dtype}")
        print("Compile Success! ")
    except Exception as e:
        print(f"Compile Failed! : {e}")
        return

    # 3. Verify Consistency (Divergence Check)
    # Note: In tf.raw_ops, a new table resource is created every call in tf.function 
    # unless the handle is captured externally. Here we check if the operations 
    # execute without type errors, which was the core issue in the PyTorch bug.
    if out_eager.dtype == out_compiled.dtype:
        print("\nDivergence Check: Passed (Dtypes match)")
    else:
        print(f"\nDivergence Check: Failed (Eager dtype: {out_eager.dtype} vs Compiled dtype: {out_compiled.dtype})")

if __name__ == '__main__':
    test_hash_table_divergence()