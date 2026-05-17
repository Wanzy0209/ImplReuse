import torch
import tensorflow as tf
from tensorflow.experimental import DynamicRaggedShape

def test_dynamic_ragged_shape_export():
    """
    Test case for tf.experimental.DynamicRaggedShape based on the logic of 
    Issue 161563 (torch.export export failure).
    
    The original issue involves exporting a complex model structure where a 
    "Current active mode" error occurs. This test adapts that logic by 
    creating a complex DynamicRaggedShape and attempting to trace/export it 
    using tf.function (TensorFlow's graph capture/export mechanism).
    """
    
    # 1. Setup: Create a complex DynamicRaggedShape
    # This mirrors the "model" and "inputs" preparation in the original issue.
    # We use a ragged tensor with varying row lengths to ensure complexity.
    rt = tf.ragged.constant([[1, 2], [3], [], [4, 5, 6]])
    shape = DynamicRaggedShape.from_tensor(rt)

    # 2. Define the Export/Trace operation
    # In PyTorch, torch.export.export captures the graph. 
    # In TensorFlow, @tf.function traces the function into a graph.
    @tf.function
    def export_shape(input_shape):
        # Accessing internal fields to verify the object behaves correctly 
        # in graph mode (analogous to the ProxyTorchDispatchMode in PyTorch).
        return input_shape.inner_shape, input_shape.row_partitions

    # 3. Execute the Export
    # The original bug raised an AssertionError here. We verify that the 
    # similar API handles the mode transition correctly.
    try:
        inner_shape, row_partitions = export_shape(shape)
    except AssertionError as e:
        # If a similar "mode not registered" error occurs here, it indicates a similar bug.
        raise AssertionError(f"Failed to export DynamicRaggedShape: {e}")

    # 4. Assertions
    # Verify that the exported graph preserves the structure correctly.
    assert isinstance(inner_shape, tf.Tensor)
    assert inner_shape.shape == (4,) # 4 rows
    
    assert len(row_partitions) == 1 # 1 ragged dimension
    assert isinstance(row_partitions[0], tf.experimental.RowPartition)
    
    # Verify the specific data integrity
    expected_splits = [0, 2, 3, 3, 6]
    assert row_partitions[0].row_splits.numpy().tolist() == expected_splits

    print("Test passed: DynamicRaggedShape exported successfully.")

if __name__ == "__main__":
    test_dynamic_ragged_shape_export()