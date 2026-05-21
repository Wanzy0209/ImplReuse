import torch
import tensorflow as tf

def test_keyvalue_tensor_initializer_type_consistency():
    """
    Test that tf.lookup.KeyValueTensorInitializer preserves tensor types
    within a tf.function context. This is analogous to the PyTorch issue where
    ReduceLROnPlateau converted a tensor LR to a float, triggering recompilation.
    Here, we ensure that the lookup operation returns a Tensor (not a Python scalar)
    to maintain graph stability and avoid retracing in tf.function.
    """
    # 1. Setup: Define keys and values as Tensors
    # Using specific dtypes to ensure type consistency is checked
    keys_tensor = tf.constant(['a', 'b', 'c'], dtype=tf.string)
    values_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

    # 2. Initialize the table using the similar API
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys_tensor, 
        values_tensor
    )
    table = tf.lookup.StaticHashTable(initializer, default_value=-1.0)

    # 3. Define a compiled function (analogous to torch.compile)
    @tf.function
    def lookup_fn(input_key):
        return table.lookup(input_key)

    # 4. Execute the function
    # Input is a tensor, output should be a tensor
    input_key = tf.constant(['a', 'b', 'x'], dtype=tf.string)
    result = lookup_fn(input_key)

    # 5. Assertions
    # The primary assertion: The result must be a Tensor.
    # If the API behaved like the buggy PyTorch scheduler, it might return a 
    # Python scalar or list, breaking the graph execution or causing retracing.
    assert isinstance(result, tf.Tensor), \
        "Result should be a Tensor to ensure graph consistency (avoiding retracing issues)."

    # Verify the dtype is preserved
    assert result.dtype == tf.float32, \
        f"Expected dtype float32, but got {result.dtype}"

    # Verify the values are correct
    expected = tf.constant([1.0, 2.0, -1.0], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), \
        "Lookup values do not match expected output."

    print("Test passed: KeyValueTensorInitializer preserves tensor types in graph mode.")

if __name__ == "__main__":
    test_keyvalue_tensor_initializer_type_consistency()