import torch

def test_as_tensor_memory_format():
    # Create a tensor with channels_last memory format
    # Using arange to match the original bug report's data generation
    x = torch.arange(0, 16).reshape(2, 2, 2, 2).to(memory_format=torch.channels_last)
    
    # Convert the tensor using torch.as_tensor
    # We explicitly cast to float to trigger a copy/conversion, 
    # ensuring we test the conversion logic rather than just reference passing.
    y = torch.as_tensor(x, dtype=torch.float32)

    # Check if the memory format (channels_last) is preserved
    is_x_channels_last = x.is_contiguous(memory_format=torch.channels_last)
    is_y_channels_last = y.is_contiguous(memory_format=torch.channels_last)
    
    print(f'Input is channels_last: {is_x_channels_last}')
    print(f'Output is channels_last: {is_y_channels_last}')
    
    # Check value equality
    values_equal = torch.equal(x, y)
    print(f'Values equal: {values_equal}')
    
    # Check stride preservation (memory layout)
    strides_match = x.stride() == y.stride()
    print(f'Strides match: {strides_match}')
    
    # Print storage to inspect memory layout details
    print(f'\nInput storage:\n{x.storage()}')
    print(f'\nOutput storage:\n{y.storage()}')

    # Assertions to verify expected behavior
    assert values_equal, "Values should be equal after conversion"
    assert strides_match, "Strides (memory format) should be preserved during conversion"
    assert is_y_channels_last, "Output should maintain channels_last memory format"

if __name__ == "__main__":
    test_as_tensor_memory_format()