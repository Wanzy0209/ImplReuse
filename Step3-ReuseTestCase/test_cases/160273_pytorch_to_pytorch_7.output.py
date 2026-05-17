import torch

def test_zeros_like_gradient_behavior():
    """
    Adapted test case for torch.zeros_like based on the gradient behavior 
    investigation of torch.min.
    
    Unlike torch.min, torch.zeros_like is a factory function that creates a new 
    tensor independent of the input tensor's values. Therefore, gradients should 
    not flow from the result back to the input tensor.
    """
    # Setup input tensor with gradient tracking
    a = torch.ones([5])
    a.requires_grad = True

    # Call the similar API: torch.zeros_like
    # We explicitly set requires_grad=True on the output to allow a backward pass,
    # mimicking the structure of the original test where the output required grad.
    z = torch.zeros_like(a, requires_grad=True)

    # Perform backward pass on the result
    z.sum().backward()

    # Verify gradient behavior
    # Since zeros_like does not depend on the values of 'a', the gradient of 'a'
    # should remain None (unlike torch.min which distributes gradients).
    assert a.grad is None, "torch.zeros_like should not pass gradients to the input tensor"
    
    # Verify the output values are correct
    assert torch.equal(z, torch.zeros([5])), "Output should be zeros"

if __name__ == "__main__":
    test_zeros_like_gradient_behavior()
    print("Test passed.")