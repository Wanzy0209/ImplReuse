import torch
import tensorflow as tf
import numpy as np

def test_variable_aggregation():
    """
    Adapted test case for tf.compat.v1.VariableAggregation.
    
    The original bug report (Issue 164086) highlights a divergence between 
    eager and compiled execution in PyTorch involving type errors with fp16.
    
    This test adapts the core logicverifying behavior consistency between 
    eager execution and compiled graphs (tf.function)to the target API 
    tf.compat.v1.VariableAggregation, using mixed precision (fp16) to 
    mirror the environment of the original bug.
    """
    
    # Setup: Enable mixed precision to align with the fp16 context of the original bug
    # This helps verify if the aggregation handles type casting correctly
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Define the aggregation types to test (equivalent to testing different op paths)
    aggregation_modes = [
        tf.compat.v1.VariableAggregation.SUM,
        tf.compat.v1.VariableAggregation.MEAN,
        tf.compat.v1.VariableAggregation.NONE,
        tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA
    ]

    # Input data: Using float16 to match the 'pointer<fp16>' context in the bug report
    initial_value = tf.constant(1.0, dtype=tf.float16)

    def run_aggregation_test(mode):
        """
        Function using the target API.
        Creates a variable with a specific aggregation mode and reads it.
        """
        # Create variable with specific aggregation
        var = tf.Variable(
            initial_value=initial_value,
            aggregation=mode,
            dtype=tf.float16,
            name=f"var_{mode.name}"
        )
        # Perform a read operation to ensure the variable is active in the graph
        return var.read_value()

    print("--- Testing Eager Execution ---")
    eager_results = {}
    for mode in aggregation_modes:
        try:
            res = run_aggregation_test(mode)
            eager_results[mode] = res
            print(f"Mode: {mode.name:<20} | Output: {res.numpy():.4f} | Dtype: {res.dtype}")
        except Exception as e:
            print(f"Mode: {mode.name:<20} | Error: {e}")

    print("\n--- Testing Compiled Execution (tf.function) ---")
    # Compile the function to check for divergences (similar to torch.compile)
    compiled_test = tf.function(run_aggregation_test)

    for mode in aggregation_modes:
        try:
            res = compiled_test(mode)
            # Verify consistency with eager execution
            if mode in eager_results:
                assert np.allclose(res.numpy(), eager_results[mode].numpy()), \
                    f"Divergence detected for {mode.name} between eager and compiled"
                assert res.dtype == eager_results[mode].dtype, \
                    f"Type mismatch for {mode.name} between eager and compiled"
            
            print(f"Mode: {mode.name:<20} | Output: {res.numpy():.4f} | Dtype: {res.dtype}")
        except Exception as e:
            print(f"Mode: {mode.name:<20} | Error: {e}")

    print("\nTest Case Finished.")

if __name__ == '__main__':
    test_variable_aggregation()