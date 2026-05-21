import torch
import tensorflow as tf

def test_enable_v2_behavior_state_access():
    """
    Test that calling enable_v2_behavior and subsequently accessing 
    internal state types (like TensorShape) does not cause a segmentation fault.
    
    This test is inspired by the PyTorch issue where accessing 
    torch.onnx.OperatorExportTypes (an enum type) during import 
    caused a segfault in pybind11's __int__ dispatcher.
    
    Here, we verify that tf.compat.v1.enable_v2_behavior correctly 
    initializes the environment and that accessing the resulting 
    state (eager execution, tensor shapes) is stable.
    """
    # Call the API that modifies global behavior, analogous to the import process in PyTorch
    tf.compat.v1.enable_v2_behavior()

    # Verify eager execution is enabled (mimics the state check)
    # In the PyTorch bug, the crash occurred when accessing the type value.
    assert tf.executing_eagerly(), "Eager execution should be enabled after calling enable_v2_behavior"

    # Access TensorShape, which is explicitly toggled by enable_v2_behavior
    # via tensor_shape.enable_v2_tensorshape(). This checks for memory safety
    # when accessing type properties (like rank, which is an int).
    shape = tf.TensorShape([1, 2, 3])
    assert shape.rank == 3, "TensorShape rank should be accessible and correct"
    
    # Accessing dimensions to further ensure type stability
    assert shape.dims == [1, 2, 3], "TensorShape dimensions should be accessible"

if __name__ == "__main__":
    test_enable_v2_behavior_state_access()
    print("Test passed: No segfault on state access after enable_v2_behavior.")