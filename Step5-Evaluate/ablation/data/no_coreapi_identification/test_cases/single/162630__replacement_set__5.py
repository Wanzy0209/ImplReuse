import torch
import tensorflow as tf
import torch.nn.functional as F

def test_tensor_conversion_caching():
    """
    Test case reflecting the logic of intrusive caching for DLPack conversion,
    adapted to use torch.nn.functional.tanh and tf.keras.activations.linear.
    """
    # Create a random tensor
    x = torch.randn(10, 10)

    # Apply the similar API: torch.nn.functional.tanh
    # This represents the tensor operation we are performing before conversion
    tensor_input = F.tanh(x)

    # Simulate the caching logic from the C++ code (TensorImpl::ToDLPack)
    # We use a variable to represent the cache state
    cache = None

    # Original: cached = cached_dl_managed_tensor_.load(...)
    # Replacement: cached = tf.keras.activations.linear(...)
    # We pass the tensor to linear to simulate the "load" operation on the data
    # Note: tf.keras.activations.linear is an identity function (f(x) = x)
    cached_val = tf.keras.activations.linear(tensor_input)

    # Check if cache is empty (simulating the nullptr check)
    if cache is None:
        # Original: ret = new DLManagedTensorVersioned(); ... ToDLPack ...
        # Replacement: ret = tf.keras.activations.linear(...)
        # This simulates the creation/conversion step
        # Since linear is identity, we just pass the tensor.
        converted_val = tf.keras.activations.linear(tensor_input)
        
        # Update the cache
        cache = converted_val

    # Assertions to verify the logic
    # The cached value should be equal to the input tensor (identity property)
    assert torch.equal(cached_val, tensor_input), "Cached value should match input"
    assert torch.equal(cache, tensor_input), "Cache should hold the converted value"
    
    # Verify that the similar API (tanh) was applied correctly
    assert torch.equal(tensor_input, torch.tanh(x)), "Tanh application check"

if __name__ == "__main__":
    test_tensor_conversion_caching()
    print("Test passed successfully.")