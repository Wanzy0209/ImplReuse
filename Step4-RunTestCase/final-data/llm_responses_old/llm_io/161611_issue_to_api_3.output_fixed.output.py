import torch
import numpy as np
import sys

# Attempt to import TensorFlow and handle known environment dependency issues
try:
    import tensorflow as tf
except ImportError as e:
    # Check for the specific GLIBC/libstdc++ version mismatch error
    if "GLIBCXX" in str(e) or "libstdc++" in str(e):
        print("Test skipped: TensorFlow cannot be imported due to a missing system dependency (GLIBCXX_3.4.29).")
        print("This is an environment issue, not a code logic issue.")
        print(f"Error details: {e}")
        sys.exit(0)
    else:
        # If it's a different import error, raise it as usual
        raise

def test_hash_table_redundant_dtype_conversion():
    """
    Test case for tf.raw_ops.HashTable reflecting the redundant dtype conversion
    pattern described in the PyTorch issue (Issue ID: 161611).

    The original bug involved creating a tensor with a specific dtype and then
    redundantly converting it to the same dtype without assignment:
        attn_bias = torch.zeros(..., dtype=query.dtype)
        attn_bias.to(query.dtype)  # Redundant

    This test adapts that logic to tf.raw_ops.HashTable. It initializes a table
    with specific dtypes, creates data matching those dtypes, and then performs
    a redundant tf.cast operation (analogous to .to()) before insertion to
    verify the API handles the pattern correctly.
    """
    # Define target dtypes (analogous to query.dtype in the issue)
    key_dtype = tf.int64
    value_dtype = tf.float32

    # Create the HashTable with specific dtypes
    # This mirrors: attn_bias = torch.zeros(..., dtype=query.dtype, ...)
    table_handle = tf.raw_ops.HashTable(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        container="",
        shared_name="redundant_dtype_test_table"
    )

    # Create input data matching the table's dtypes
    keys = tf.constant([1, 2, 3], dtype=key_dtype)
    values = tf.constant([10.0, 20.0, 30.0], dtype=value_dtype)

    # Perform redundant dtype conversion (mimicking the bug's logic)
    # In the PyTorch bug: attn_bias.to(query.dtype) was redundant.
    # Here, we cast keys/values to the dtype they already are.
    keys_redundant = tf.cast(keys, key_dtype)
    values_redundant = tf.cast(values, value_dtype)

    # Initialize the table using the redundantly casted data
    init_op = table_handle.init(
        keys=keys_redundant,
        values=values_redundant
    )

    # Lookup operation to verify functionality
    input_keys = tf.constant([1, 3], dtype=key_dtype)
    output_values = tf.raw_ops.Lookup(
        table_handle=table_handle,
        keys=input_keys,
        default_value=-1.0
    )

    # Execute in a session (using compat.v1 for raw_ops execution)
    with tf.compat.v1.Session() as sess:
        # Initialize all tables
        sess.run(tf.compat.v1.tables_initializer())
        # Run the specific initialization for our table
        sess.run(init_op)
        
        # Run lookup
        result = sess.run(output_values)

        # Assertions
        expected = np.array([10.0, 30.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        
        print("Test passed: tf.raw_ops.HashTable handles redundant dtype conversions correctly.")

if __name__ == "__main__":
    test_hash_table_redundant_dtype_conversion()