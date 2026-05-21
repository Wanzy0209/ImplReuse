import torch
import tensorflow as tf
import numpy as np

def test_tf_convert_to_tensor_complex_graph():
    """
    Adapts the PyTorch fuzzer crash case to TensorFlow.
    The original issue involved a complex graph of matmul operations causing
    a guard failure in torch._dynamo. This test verifies that tf.convert_to_tensor
    can correctly handle the input types and shapes required to construct a similar
    graph in TensorFlow, and that the graph executes successfully (including compilation).
    """
    # Set seed for reproducibility
    np.random.seed(52676)

    # Generate input data using numpy (simulating external inputs)
    # Shapes are derived from the comments in the original PyTorch code
    arg_0 = np.random.randn(9, 9, 9).astype(np.float64)
    arg_1 = np.random.randn(9, 9, 11).astype(np.float64)
    arg_2 = np.random.randn(9, 12, 8).astype(np.float64)
    arg_3 = np.random.randn(9, 8, 13).astype(np.float64)
    arg_4 = np.random.randn(9, 13, 7).astype(np.float64)
    arg_5 = np.random.randn(9, 7, 16).astype(np.float64)
    arg_6 = np.random.randn(9, 16, 12).astype(np.float64)
    arg_7 = np.random.randn(9, 12, 11).astype(np.float64)
    
    # Sentinel argument present in original signature
    sentinel = None

    # Define the computation logic
    # We use @tf.function to mimic the compilation aspect of torch._dynamo
    @tf.function
    def fuzzed_program_tf(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
        # Use tf.convert_to_tensor (the API under test) to ensure inputs are valid tensors
        # This tests the conversion logic for the specific shapes/dtypes involved in the bug
        var_node_6 = tf.convert_to_tensor(arg_0, dtype=tf.float64)
        var_node_7 = tf.convert_to_tensor(arg_1, dtype=tf.float64)
        
        var_node_5 = tf.matmul(var_node_6, var_node_7)
        
        # torch.full equivalent
        var_node_9 = tf.constant(1.5758497316910556, shape=(9, 11, 12), dtype=tf.float64)
        
        var_node_10 = tf.convert_to_tensor(arg_2, dtype=tf.float64)
        var_node_8 = tf.matmul(var_node_9, var_node_10)
        
        var_node_4 = tf.matmul(var_node_5, var_node_8)
        
        var_node_13 = tf.convert_to_tensor(arg_3, dtype=tf.float64)
        var_node_14 = tf.convert_to_tensor(arg_4, dtype=tf.float64)
        var_node_12 = tf.matmul(var_node_13, var_node_14)
        
        var_node_15 = tf.convert_to_tensor(arg_5, dtype=tf.float64)
        var_node_11 = tf.matmul(var_node_12, var_node_15)
        
        var_node_3 = tf.matmul(var_node_4, var_node_11)
        
        var_node_17 = tf.convert_to_tensor(arg_6, dtype=tf.float64)
        var_node_18 = tf.convert_to_tensor(arg_7, dtype=tf.float64)
        var_node_16 = tf.matmul(var_node_17, var_node_18)
        
        var_node_2 = tf.matmul(var_node_3, var_node_16)
        
        # torch.full equivalent
        var_node_23 = tf.constant(-0.5249394453404403, shape=(156, 8), dtype=tf.float64)
        var_node_24 = tf.constant(0.9331226188585692, shape=(8, 9), dtype=tf.float64)
        var_node_22 = tf.matmul(var_node_23, var_node_24)
        
        # torch.full equivalent
        var_node_26 = tf.constant(-0.9276381954691514, shape=(9, 13), dtype=tf.float64)
        
        return var_node_2, var_node_22, var_node_26

    # Execute the function
    try:
        result = fuzzed_program_tf(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel)
        
        # Basic assertions to verify execution and shape preservation
        assert result[0].shape == (9, 9, 11), f"Shape mismatch for node_2: {result[0].shape}"
        assert result[1].shape == (156, 9), f"Shape mismatch for node_22: {result[1].shape}"
        assert result[2].shape == (9, 13), f"Shape mismatch for node_26: {result[2].shape}"
        
        print("Test passed: tf.convert_to_tensor successfully handled inputs for the complex graph.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_tf_convert_to_tensor_complex_graph()