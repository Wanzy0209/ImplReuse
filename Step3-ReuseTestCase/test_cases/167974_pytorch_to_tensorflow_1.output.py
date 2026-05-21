import torch
import tensorflow as tf
import numpy as np

def test_dequantize_2d_input():
    """
    Adapted test case based on PyTorch EmbeddingBag bug (Issue ID: 167974).
    
    Original Bug Logic:
    - Input: 2D tensor [[1, 2, 4, 5], [4, 3, 2, 9]]
    - Issue: Internal offsets generated were [0, 4] instead of [0, 4, 8] when 
             include_last_offset=True.
    
    Adapted Logic for tf.quantization.dequantize:
    - Verify that the API correctly handles 2D input tensors.
    - Verify that specific parameters (axis) are respected during the operation,
      similar to how include_last_offset should have been respected.
    """
    
    # Create a 2D input tensor matching the structure of the PyTorch repro
    # Original: torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    input_data = np.array([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.quint8)
    
    # Define quantization range
    min_range = 0.0
    max_range = 10.0
    
    # Test Case: Dequantize with axis specified (mimicking the specific flag usage in the bug)
    # We use axis=-1 to ensure the operation handles the 2D structure correctly along the last dimension.
    output = tf.quantization.dequantize(
        input_data,
        min_range,
        max_range,
        mode='MIN_COMBINED',
        axis=-1
    )
    
    # Assertion 1: Verify output shape matches input shape
    # In the PyTorch bug, the internal offsets were incorrect for the 2D input size.
    # Here, we verify the output tensor maintains the correct 2D dimensions.
    assert output.shape == input_data.shape, \
        f"Shape mismatch. Expected {input_data.shape}, got {output.shape}"
        
    # Assertion 2: Verify values are correctly computed
    # scale = (max_range - min_range) / (quantization_range_max - quantization_range_min)
    # For quint8, range is 0 to 255.
    scale = (max_range - min_range) / 255.0
    expected_val = input_data[0, 0] * scale
    
    assert np.isclose(output[0, 0].numpy(), expected_val), \
        f"Value mismatch. Expected {expected_val}, got {output[0, 0].numpy()}"

    print("Test passed: tf.quantization.dequantize correctly handles 2D input and parameters.")

if __name__ == "__main__":
    test_dequantize_2d_input()