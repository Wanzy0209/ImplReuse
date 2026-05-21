import torch
import torch.nn.functional as F

def test_linear_share_memory():
    """
    Test case to verify that the output of torch.nn.functional.linear
    supports share_memory_() without crashing, similar to the reported
    issue with NestedTensor.
    """
    # Create inputs for torch.nn.functional.linear
    # Input shape: (Batch size, Input features)
    input_tensor = torch.randn(10, 5)
    # Weight shape: (Output features, Input features)
    weight_tensor = torch.randn(3, 5)
    # Bias shape: (Output features)
    bias_tensor = torch.randn(3)

    # Call the similar API
    output = F.linear(input_tensor, weight_tensor, bias_tensor)

    # Adapt the original call site (share_memory_) to verify the similar API
    # The original bug was a segmentation fault, so we check if this runs without error.
    try:
        output.share_memory_()
    except Exception as e:
        print(f"share_memory_ failed with exception: {e}")
        raise

    # Verify that the memory is actually shared
    assert output.is_shared(), "Output tensor should be in shared memory after calling share_memory_()"

if __name__ == "__main__":
    test_linear_share_memory()
    print("Test passed: torch.nn.functional.linear output supports share_memory_.")