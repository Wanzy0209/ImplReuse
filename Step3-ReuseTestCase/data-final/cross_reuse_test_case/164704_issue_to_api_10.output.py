import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_ops_exp_int16_divergence():
    """
    Test case for tf.keras.ops.exp based on PyTorch Issue 164704.
    
    The original issue involves a divergence between eager and compiled modes 
    in PyTorch when handling scalar int16 tensors, specifically related to 
    type expectations (int vs float). 
    
    This test adapts the input generation logic (int16 tensors, unique, squeeze)
    to verify that tf.keras.ops.exp handles the type promotion (int16 -> float)
    consistently between eager execution and tf.function (compiled) execution.
    """
    
    # Replicate the tensor setup from the PyTorch bug
    # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
    var_node_4 = tf.constant(3, shape=(2, 3), dtype=tf.int16)

    # var_node_3 = torch.unique(var_node_4)
    # PyTorch unique flattens the input. We must do the same for tf.unique.
    flat_4 = tf.reshape(var_node_4, [-1])
    var_node_3 = tf.unique(flat_4).y

    # var_node_2 = torch.squeeze(var_node_3)
    # This results in a scalar tensor of dtype int16
    var_node_2 = tf.squeeze(var_node_3)

    # Define the operation using the Similar API: tf.keras.ops.exp
    # We expect this to promote int16 to float32.
    def run_op(x):
        return tf.keras.ops.exp(x)

    # 1. Test Eager Execution
    print("Testing Eager Execution...")
    try:
        result_eager = run_op(var_node_2)
        print(f"  Eager Result: {result_eager.numpy()}, Dtype: {result_eager.dtype}")
        # tf.exp typically promotes int to float32
        assert result_eager.dtype == tf.float32, "Eager mode should promote int16 to float32"
    except Exception as e:
        print(f"   Eager execution failed: {e}")
        raise

    # 2. Test Compiled Execution (tf.function)
    print("Testing Compiled Execution...")
    try:
        compiled_op = tf.function(run_op)
        result_compiled = compiled_op(var_node_2)
        print(f"  Compiled Result: {result_compiled.numpy()}, Dtype: {result_compiled.dtype}")
        assert result_compiled.dtype == tf.float32, "Compiled mode should promote int16 to float32"
    except Exception as e:
        print(f"   Compiled execution failed: {e}")
        raise

    # 3. Check for Divergence
    print("Checking for Divergence...")
    assert np.allclose(result_eager.numpy(), result_compiled.numpy()), \
        "Value divergence detected between eager and compiled modes"
    assert result_eager.dtype == result_compiled.dtype, \
        "Dtype divergence detected between eager and compiled modes"

    print(" Test passed: No eager/compile divergence for tf.keras.ops.exp with int16 inputs.")

if __name__ == "__main__":
    test_tf_keras_ops_exp_int16_divergence()