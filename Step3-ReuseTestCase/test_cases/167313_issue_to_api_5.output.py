import torch
import tensorflow as tf

def test_keyvalue_initializer_params():
    """
    Test case adapted from PyTorch Issue 167313.
    
    Original Bug: torch.addmm ignored alpha/beta parameters when compiled.
    Adaptation: Verify that tf.lookup.KeyValueTensorInitializer respects 
    key_dtype/value_dtype parameters when used inside tf.function (compiled).
    """
    
    # Setup inputs (using python lists to allow dtype specification via initializer)
    keys = [1, 2, 3]
    values = [10, 20, 30]
    query = tf.constant([1, 2], dtype=tf.int64)

    # Define function using the similar API with optional parameters
    # Analogous to: torch.addmm(x, a, b, alpha=0.5, beta=0.5)
    # Here we test key_dtype and value_dtype
    f = lambda k, v: tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            k, v,
            key_dtype=tf.int64,   # Optional parameter 1
            value_dtype=tf.float32 # Optional parameter 2
        ),
        default_value=-1.0
    ).lookup(query)

    # Compile the function (Analogous to torch.compile)
    fc = tf.function(f)

    # Execute eager
    result_eager = f(keys, values)

    # Execute compiled
    result_compiled = fc(keys, values)

    # Assertions
    # 1. Check consistency between eager and compiled (mirroring the bug report's comparison)
    assert tf.reduce_all(result_eager == result_compiled).numpy(), "Mismatch between eager and compiled results"

    # 2. Check if the optional parameters (dtypes) were respected
    # If the initializer ignored parameters like the PyTorch bug, this might fail or behave unexpectedly
    assert result_eager.dtype == tf.float32, f"Expected float32, got {result_eager.dtype}"
    
    # 3. Verify values are correct based on the types
    expected_values = tf.constant([10.0, 20.0], dtype=tf.float32)
    assert tf.reduce_all(result_eager == expected_values).numpy(), "Values do not match expected output"

    print("Test passed. Parameters respected in both eager and compiled modes.")

if __name__ == "__main__":
    test_keyvalue_initializer_params()