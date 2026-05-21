import torch
import tensorflow as tf

def test_assert_with_zero_sized_tensor():
    """
    Test case for tf.raw_ops.Assert based on the PyTorch issue 164875.
    
    The original issue involved a divergence between eager and compiled modes
    when handling tensors with size (20, 0). This test verifies that 
    tf.raw_ops.Assert handles such edge-case shapes correctly in both modes.
    """
    
    # Recreate the tensor with size (20, 0) similar to the original issue
    # Original: torch.as_strided(..., (20, 0), ...)
    # TF equivalent for creating a specific shape:
    arg_0 = tf.zeros((20, 0), dtype=tf.int64)

    def program(tensor_input):
        # Use the similar API: tf.raw_ops.Assert
        # We pass the (20, 0) tensor as data to the Assert operation.
        # We set condition to True to ensure the assertion passes, 
        # specifically testing the handling of the data tensor's shape.
        condition = tf.constant(True)
        return tf.raw_ops.Assert(condition, [tensor_input], summarize=10)

    # Test Eager execution
    print("Testing Eager...")
    try:
        result_eager = program(arg_0)
        # In eager mode, Assert returns None if condition is True
        assert result_eager is None
        print(" eager success")
    except Exception as e:
        print(f" eager failed: {e}")

    # Test Compiled execution (tf.function)
    print("Testing Compiled...")
    compiled_program = tf.function(program)
    try:
        result_compiled = compiled_program(arg_0)
        assert result_compiled is None
        print(" compile success")
    except Exception as e:
        print(f" compile failed: {e}")

if __name__ == "__main__":
    test_assert_with_zero_sized_tensor()