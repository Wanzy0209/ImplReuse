import torch
import tensorflow as tf

# This test case adapts the logic from the PyTorch issue (verifying that a 
# specific configuration argument is respected by the API) to the 
# tf.VariableSynchronization API.

def test_variable_synchronization_config():
    """
    Test that the synchronization configuration passed to tf.Variable
    is correctly preserved, similar to how dynamic_axes names should be
    preserved in torch.onnx.export.
    """
    
    # Define the desired synchronization behavior
    # This mirrors the 'dynamic_axes' dictionary definition in the original issue
    target_sync = tf.VariableSynchronization.ON_READ
    target_agg = tf.VariableAggregation.MEAN

    # Create a variable with the specific configuration
    # This corresponds to the torch.onnx.export call with arguments
    var = tf.Variable(
        [0.],
        trainable=False,
        synchronization=target_sync,
        aggregation=target_agg
    )

    # Verify that the variable instance reflects the configuration provided
    # This corresponds to checking onnx_model.graph.input/output in the original issue
    assert var.synchronization == target_sync, (
        f"Expected synchronization to be {target_sync}, "
        f"but got {var.synchronization}"
    )
    
    assert var.aggregation == target_agg, (
        f"Expected aggregation to be {target_agg}, "
        f"but got {var.aggregation}"
    )

    print("Test passed: VariableSynchronization configuration is correctly applied.")

if __name__ == "__main__":
    test_variable_synchronization_config()