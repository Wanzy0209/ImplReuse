import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_exp_divergence():
    """
    Test case adapted from PyTorch Issue 164704.
    Original issue: RuntimeError: expected int arg but got float during torch.compile.
    This test checks for similar eager/compile divergence or type handling issues
    when using tf.keras.backend.exp in a context involving integer tensor reductions.
    """
    
    # Setup inputs mimicking the original bug's int16 tensors
    # Original: arg_0 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
    arg_0 = tf.constant(10, dtype=tf.int16)
    arg_1 = tf.constant(5, dtype=tf.int16)

    # Define the logic using the similar API (tf.keras.backend.exp)
    # The original logic involved: unique -> squeeze -> sub -> add -> div -> item -> mul
    # We adapt this to use exp to test the API in a similar context.
    def logic_fn(x, y):
        # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
        var_node_4 = tf.fill((2, 3), tf.cast(3, tf.int16))
        
        # var_node_3 = torch.unique(var_node_4)
        # tf.unique returns (unique, indices) in newer versions, handling 1D input
        var_node_3, _ = tf.unique(tf.reshape(var_node_4, [-1]))
        
        # var_node_2 = torch.squeeze(var_node_3)
        var_node_2 = tf.squeeze(var_node_3)
        
        # var_node_6 = torch.sub(arg_0, arg_1)
        var_node_6 = tf.subtract(x, y)
        
        # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
        var_node_10 = tf.fill((1,), tf.cast(3, tf.int16))
        # var_node_9 = torch.squeeze(var_node_10)
        var_node_9 = tf.squeeze(var_node_10)
        
        # var_node_5 = torch.add(var_node_6, var_node_9)
        var_node_5 = tf.add(var_node_6, var_node_9)
        
        # var_node_1 = torch.div(var_node_2, var_node_5)
        # Casting to float is necessary for division in TF to avoid integer division truncation
        # which might differ from PyTorch's behavior depending on version, 
        # but here we want to feed into exp.
        var_node_div = tf.cast(var_node_2, tf.float32) / tf.cast(var_node_5, tf.float32)
        
        # Use the similar API: tf.keras.backend.exp
        # This replaces the item() -> scalar multiplication logic from the original bug
        # to test if exp handles the scalar/tensor type transition correctly in graph mode.
        result = tf.keras.backend.exp(var_node_div)
        
        return result

    # 1. Test Eager Mode
    eager_result = logic_fn(arg_0, arg_1)
    print(f"Eager Result: {eager_result.numpy()}, Dtype: {eager_result.dtype}")

    # 2. Test Compiled Mode (tf.function)
    # This mimics torch.compile in the original issue
    compiled_fn = tf.function(logic_fn)
    compiled_result = compiled_fn(arg_0, arg_1)
    print(f"Compiled Result: {compiled_result.numpy()}, Dtype: {compiled_result.dtype}")

    # 3. Assertions
    # Check for divergence (values should match)
    assert np.allclose(eager_result.numpy(), compiled_result.numpy()), \
        f"Divergence detected: Eager={eager_result.numpy()}, Compiled={compiled_result.numpy()}"
    
    # Check for type consistency (The original bug was "expected int but got float")
    assert eager_result.dtype == compiled_result.dtype, \
        f"Dtype divergence: Eager={eager_result.dtype}, Compiled={compiled_result.dtype}"
        
    # tf.keras.backend.exp is expected to return float
    assert eager_result.dtype == tf.float32, \
        f"Expected float32 output from exp, got {eager_result.dtype}"

    print(" Test passed: No eager/compile divergence detected.")

if __name__ == "__main__":
    test_tf_keras_exp_divergence()