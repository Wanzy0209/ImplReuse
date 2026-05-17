import torch

def test_torch_min_gradient_behavior():
    """
    Test to verify the gradient behavior of torch.min based on Issue 160273.
    
    Behavior 1: torch.min(input) reduces over all dimensions.
    Expected: Gradients are evenly distributed among equal minimum values.
    
    Behavior 2: torch.min(input, dim=...) reduces over a specified dimension.
    Expected: Gradients are NOT evenly distributed (acts like indexing/argmin).
    """
    
    # --- Test Case 1: Global reduction (torch.min(input)) ---
    # Create a tensor of ones where all elements are equal
    a = torch.ones([5], requires_grad=True)
    min_val = torch.min(a)
    min_val.backward()
    
    # Since all 5 elements are the minimum, the gradient (1.0) should be split evenly (1/5 = 0.2)
    expected_grad_global = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    assert torch.allclose(a.grad, expected_grad_global), \
        f"Global min gradient failed. Expected {expected_grad_global}, got {a.grad}"

    # --- Test Case 2: Dimensional reduction (torch.min(input, dim=...)) ---
    # Create a tensor of ones where all elements are equal
    b = torch.ones([5], requires_grad=True)
    min_val_dim = torch.min(b, dim=0)
    
    # Backpropagate on the values tensor returned by the dimensional reduction
    min_val_dim.values.backward()
    
    # The gradient flows back to the first index of the minimum value (indexing behavior)
    expected_grad_dim = torch.tensor([1., 0., 0., 0., 0.])
    assert torch.allclose(b.grad, expected_grad_dim), \
        f"Dimensional min gradient failed. Expected {expected_grad_dim}, got {b.grad}"

def test_torch_max_gradient_behavior():
    """
    Test to verify the gradient behavior of torch.max based on Issue 160273.
    Should mirror the behavior of torch.min.
    """
    
    # --- Test Case 3: Global reduction (torch.max(input)) ---
    c = torch.ones([5], requires_grad=True)
    max_val = torch.max(c)
    max_val.backward()
    
    expected_grad_global = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    assert torch.allclose(c.grad, expected_grad_global), \
        f"Global max gradient failed. Expected {expected_grad_global}, got {c.grad}"

    # --- Test Case 4: Dimensional reduction (torch.max(input, dim=...)) ---
    d = torch.ones([5], requires_grad=True)
    max_val_dim = torch.max(d, dim=0)
    max_val_dim.values.backward()
    
    expected_grad_dim = torch.tensor([1., 0., 0., 0., 0.])
    assert torch.allclose(d.grad, expected_grad_dim), \
        f"Dimensional max gradient failed. Expected {expected_grad_dim}, got {d.grad}"

if __name__ == "__main__":
    test_torch_min_gradient_behavior()
    test_torch_max_gradient_behavior()
    print("All tests passed.")