import torch

def test_index_select_gradient():
    """
    Test case for torch.index_select gradient behavior.
    Adapted from the torch.min issue (ID: 160273) to verify that index_select
    behaves like the "indexed" gradient mode (gradients flow only to selected indices),
    similar to torch.min(input, dim=...).
    """
    # Setup: Create a tensor of ones on CUDA (matching the original issue context)
    a = torch.ones([5]).cuda()
    a.requires_grad = True

    # Operation: Select the first element (index 0)
    # This mimics the behavior of torch.min(a, dim=0) on a tensor of ones,
    # where the gradient is assigned to the first index found.
    index = torch.tensor([0]).cuda()
    selected_val = torch.index_select(a, 0, index)

    # Backward pass
    selected_val.backward()

    # Verification: Check that only the selected index received the gradient.
    # Expected: [1., 0., 0., 0., 0.]
    expected_grad = torch.tensor([1., 0., 0., 0., 0.], device='cuda:0')
    
    assert torch.allclose(a.grad, expected_grad), f"Gradient mismatch: expected {expected_grad}, got {a.grad}"
    print("Test passed: torch.index_select gradient behaves as expected.")

if __name__ == "__main__":
    test_index_select_gradient()