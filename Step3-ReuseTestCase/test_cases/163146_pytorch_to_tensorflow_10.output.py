import torch
import tensorflow as tf
import numpy as np

def test_spatial_dropout_with_dynamic_slicing():
    """
    Adapted test case for PyTorch Issue #163146.
    
    Original Bug: torch.export.export failed with a "Data dependent error" 
    when encountering dynamic slicing logic: `item_embedding[:, :max_item_num, :]`.
    
    This test verifies that the TensorFlow equivalent API (tf.keras.layers.SpatialDropout1D)
    handles dynamic slicing and dynamic input shapes correctly within a graph context
    (tf.function), which is analogous to torch.export.
    """
    
    # 1. Initialize the API under test
    # SpatialDropout1D drops entire 1D feature maps along the timestep dimension.
    rate = 0.5
    layer = tf.keras.layers.SpatialDropout1D(rate)

    # 2. Define the model logic including the problematic slicing pattern
    # We use tf.function to simulate the graph export/tracing process.
    @tf.function(input_signature=[
        # Input tensor with dynamic batch, sequence, and feature dimensions
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="item_embedding"),
        # Scalar tensor for the dynamic slice limit
        tf.TensorSpec(shape=[], dtype=tf.int32, name="max_item_num")
    ])
    def model_logic(item_embedding, max_item_num):
        # Reproduce the core logic from the bug report:
        # "selected_item_embedding = item_embedding[:, :max_item_num, :]"
        # This operation caused the "Data dependent error" in PyTorch export.
        selected_item_embedding = item_embedding[:, :max_item_num, :]
        
        # Apply the layer under test
        output = layer(selected_item_embedding, training=True)
        return output

    # 3. Prepare test data
    # Shape: [Batch=2, Seq=10, Features=64]
    # Corresponds to the 'item_embedding' in the bug report
    batch_size = 2
    seq_len = 10
    features = 64
    dummy_input = np.random.rand(batch_size, seq_len, features).astype(np.float32)
    
    # Dynamic slice limit
    # Corresponds to 'max_item_num' in the bug report
    slice_limit = 5

    # 4. Execute the graph
    try:
        # Convert inputs to tensors
        input_tensor = tf.constant(dummy_input)
        limit_tensor = tf.constant(slice_limit, dtype=tf.int32)

        # Run the traced function
        result = model_logic(input_tensor, limit_tensor)

        # 5. Assertions
        # Verify the slicing worked: Sequence dimension should be reduced to slice_limit
        expected_shape = [batch_size, slice_limit, features]
        
        assert result.shape == expected_shape, (
            f"Shape mismatch after dynamic slicing and layer application. "
            f"Expected {expected_shape}, got {result.shape.as_list()}"
        )
        
        # Verify the output is a tensor (sanity check)
        assert isinstance(result, tf.Tensor), "Output is not a Tensor"

        print("Test Passed: SpatialDropout1D handles dynamic slicing and shapes correctly.")

    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_spatial_dropout_with_dynamic_slicing()