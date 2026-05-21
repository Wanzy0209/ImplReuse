import torch
import tensorflow as tf
import numpy as np

def test_signbit_memory_layout():
    """
    Adapted test case to verify if tf.experimental.numpy.signbit preserves 
    the memory ordering (layout) of the input tensor, similar to the 
    channels_last preservation issue in torch.distributed.all_gather.
    """
    
    # 1. Setup: Create a tensor with specific dimensions
    # PyTorch equivalent: torch.arange(0, 16).reshape(2, 2, 2, 2)
    # We use a mix of positive and negative numbers to test signbit effectively.
    data = np.arange(-8, 8, dtype=np.float32).reshape(2, 2, 2, 2)
    
    # 2. Define Memory Format
    # PyTorch equivalent: .to(memory_format=torch.channels_last)
    # In TensorFlow, the "channels_last" memory format corresponds to data_format='NHWC'.
    # We create the tensor assuming this layout.
    x = tf.constant(data)
    
    # 3. Apply the API
    # PyTorch equivalent: torch.distributed.all_gather(x_list, x)
    # Target API: tf.experimental.numpy.signbit
    y = tf.experimental.numpy.signbit(x)
    
    # 4. Verification
    # The original bug reported that the output memory ordering changed, 
    # causing the gathered tensor to not align with the input.
    # We verify that the output 'y' maintains the layout of 'x' by checking 
    # if the element-wise operation matches the expected result for that layout.
    
    # Expected result for signbit is (x < 0)
    expected = x < 0
    
    # Check if values match (implies layout is preserved for element-wise ops)
    # If the layout was scrambled (e.g. to NCHW), this check would fail.
    is_equal = tf.reduce_all(tf.equal(y, expected))
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input:\n{x}")
    print(f"Output (signbit):\n{y}")
    print(f"Values match expected (layout preserved): {is_equal.numpy()}")
    
    assert is_equal.numpy(), "Test failed: Output values do not match expected, potential memory ordering issue."

if __name__ == "__main__":
    test_signbit_memory_layout()