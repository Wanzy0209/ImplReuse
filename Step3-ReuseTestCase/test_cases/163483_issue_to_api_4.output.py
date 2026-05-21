import torch
import math

def test_exp2_memory_format_preservation():
    """
    Test case to verify if torch.special.exp2 preserves the memory format 
    (channels_last) of the input tensor, similar to the issue reported 
    for torch.distributed.all_gather.
    """
    # Create a 4D tensor suitable for channels_last memory format
    x = torch.arange(0, 16, dtype=torch.float32).reshape(2, 2, 2, 2)
    
    # Convert to channels_last memory format
    x_cl = x.to(memory_format=torch.channels_last)
    
    # Apply the similar API: torch.special.exp2
    y = torch.special.exp2(x_cl)
    
    # Verify the memory format is preserved
    # The original bug reported that all_gather changed the memory ordering,
    # causing misalignment. We check if exp2 maintains the channels_last format.
    assert y.is_contiguous(memory_format=torch.channels_last), \
        "torch.special.exp2 did not preserve channels_last memory format"
    
    # Verify the values are computed correctly
    expected = torch.exp2(x)
    assert torch.allclose(y, expected), "Output values are incorrect"

    print("Test passed: torch.special.exp2 preserves channels_last memory format.")

if __name__ == "__main__":
    test_exp2_memory_format_preservation()