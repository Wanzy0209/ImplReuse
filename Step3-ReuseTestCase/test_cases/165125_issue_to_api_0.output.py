import torch
import tensorflow as tf

def test_hash_table_init_return_type():
    """
    Test case for tf.lookup.HashTable.init based on the similarity to the
    torch.utils.cpp_extension._jit_compile bug.

    The original bug reported that _jit_compile was typed as returning None
    but actually returned a module or str. This test verifies that the
    similar API (HashTable.init) correctly returns a non-None value
    (specifically a tf.Operation), ensuring the return type is handled correctly.
    """
    # Create a simple HashTable
    keys = tf.constant([1, 2, 3], dtype=tf.int64)
    vals = tf.constant(["a", "b", "c"], dtype=tf.string)
    table = tf.lookup.HashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), 
        default_value=""
    )

    # Get the return value of the 'init' property
    initializer = table.init

    # The original bug was about a return type mismatch (None vs actual).
    # We assert that the return value is not None and is the expected type.
    assert initializer is not None, "HashTable.init should not return None"
    assert isinstance(initializer, tf.Operation), \
        f"HashTable.init should return a tf.Operation, got {type(initializer)}"

if __name__ == "__main__":
    test_hash_table_init_return_type()
    print("Test passed.")