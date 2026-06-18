import torch
import tensorflow as tf
import numpy as np

def test_tf_scalar_tensor_mul_compilation():
    """
    Translates the PyTorch bug reproduction logic to TensorFlow.
    The original bug involves a divergence between eager and compiled modes
    when multiplying a Python scalar (extracted from a tensor) with a tensor.
    
    Original API: torch.mul
    Similar API: tf.keras.optimizers.serialize (Context: TensorFlow library)
    """
    
    # Setup inputs similar to the PyTorch issue
    # sentinel: float tensor (requires_grad equivalent in TF is just watching it, 
    # but for the operation itself, a float tensor is sufficient)
    sentinel = tf.constant(1.0)
    
    # arg_0: bool tensor size=(1,)
    arg_0 = tf.constant([True], dtype=tf.bool)

    def fuzzed_program(arg_0, sentinel):
        # var_node_1 = torch.squeeze(var_node_2) -> size=()
        var_node_1 = tf.squeeze(arg_0)
        
        # var_node_0 = var_node_1.item() -> dtype=bool (Python scalar)
        # In TensorFlow, extracting a Python scalar inside a tf.function 
        # typically requires tf.py_function or similar mechanisms to break the graph.
        def get_item(x):
            return x.numpy().item()
        
        # We use tf.py_function to simulate the .item() call which returns a Python scalar
        var_node_0 = tf.py_function(get_item, [var_node_1], tf.bool)
        
        # result = var_node_0 * sentinel
        # This is the critical operation: Scalar (bool) * Tensor (float)
        result = var_node_0 * sentinel
        
        return result

    # 1. Eager Execution
    result_eager = fuzzed_program(arg_0, sentinel)
    print(f"Eager result: {result_eager.numpy()}")

    # 2. Compiled Execution (tf.function equivalent to torch.compile)
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(arg_0, sentinel)
    print(f"Compiled result: {result_compiled.numpy()}")

    # 3. Assertion for Divergence
    # The PyTorch bug resulted in a TypeError during compilation.
    # Here we assert that TF handles this consistently or raises an expected error.
    try:
        # Check if results match
        assert np.array_equal(result_eager.numpy(), result_compiled.numpy()), \
            "Divergence detected: Eager and Compiled results differ."
        print(" Test passed: No divergence between eager and compiled modes.")
    except Exception as e:
        print(f" Test failed: {e}")
        raise

if __name__ == "__main__":
    test_tf_scalar_tensor_mul_compilation()