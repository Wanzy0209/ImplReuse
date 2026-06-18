import torch
import tensorflow as tf

def test_tf_exp_int16_divergence():
    """
    Test case for tf.compat.v1.math.exp based on PyTorch issue 164704.
    The original issue involved a divergence between eager and compiled modes
    when handling scalar types (int16 vs float) after tensor operations.
    This test replicates the logic flow using tf.compat.v1.math.exp to check
    for similar type handling issues in TensorFlow's graph compilation.
    """
    
    # Setup inputs similar to the PyTorch bug report
    # PyTorch: arg_0 = torch.as_strided(..., dtype=torch.int16)
    arg_0 = tf.constant(10, dtype=tf.int16)
    arg_1 = tf.constant(5, dtype=tf.int16)
    
    # Sentinel tensor to ensure gradient/float interaction
    sentinel = tf.constant(1.0, dtype=tf.float32)

    def fuzzed_program(arg_0, arg_1, sentinel):
        # Replicate the tensor manipulation logic from the bug report
        # PyTorch: var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
        var_node_4 = tf.fill((2, 3), tf.cast(3, tf.int16))
        
        # PyTorch: var_node_3 = torch.unique(var_node_4)
        # PyTorch: var_node_2 = torch.squeeze(var_node_3)
        # In TF, unique flattens, but if all elements are same, unique returns [val].
        # Squeezing [val] gives a scalar.
        var_node_3, _ = tf.unique(var_node_4)
        var_node_2 = tf.squeeze(var_node_3) # scalar int16

        # Setup second operand
        var_node_7 = arg_0
        var_node_8 = arg_1
        var_node_6 = tf.subtract(var_node_7, var_node_8)
        var_node_10 = tf.fill((1,), tf.cast(3, tf.int16))
        var_node_9 = tf.squeeze(var_node_10)
        var_node_5 = tf.add(var_node_6, var_node_9) # scalar int16

        # Original PyTorch: var_node_1 = torch.div(var_node_2, var_node_5)
        # Similar API: tf.compat.v1.math.exp
        # We apply exp to the int16 scalar. Note: tf.exp promotes int to float.
        # This tests if the compiler handles the type promotion correctly.
        var_node_1 = tf.compat.v1.math.exp(var_node_2)

        # PyTorch: var_node_0 = var_node_1.item()
        # In TF, var_node_1 is a 0-d tensor (scalar).
        var_node_0 = var_node_1

        # Arithmetic with float sentinel
        result = var_node_0 * sentinel
        return result

    # 1. Eager Execution
    try:
        result_eager = fuzzed_program(arg_0, arg_1, sentinel)
        print(f" Eager success: {result_eager.numpy()}, dtype: {result_eager.dtype}")
    except Exception as e:
        print(f" Eager failure: {e}")
        return

    # 2. Compiled Execution (tf.function)
    # This is equivalent to torch.compile in the bug report
    compiled_program = tf.function(fuzzed_program)
    try:
        result_compiled = compiled_program(arg_0, arg_1, sentinel)
        print(f" Compile success: {result_compiled.numpy()}, dtype: {result_compiled.dtype}")
    except Exception as e:
        print(f" Compile failure: {e}")
        return

    # 3. Check for Divergence
    # The original bug was a RuntimeError due to type mismatch.
    # We check if values and dtypes match.
    if result_eager.dtype != result_compiled.dtype:
        print(f" Dtype Divergence: Eager={result_eager.dtype}, Compiled={result_compiled.dtype}")
    elif not tf.reduce_all(tf.equal(result_eager, result_compiled)).numpy():
        print(f" Value Divergence: Eager={result_eager.numpy()}, Compiled={result_compiled.numpy()}")
    else:
        print(" No divergence detected.")

if __name__ == "__main__":
    test_tf_exp_int16_divergence()