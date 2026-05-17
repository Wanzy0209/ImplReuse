import torch

def test_torch_cumprod_gradient():
    # torch.cumprod does not reduce dimensions, so we sum the output to perform backward pass.
    
    # Case 1: 1D tensor, cumprod over dim=0
    a = torch.ones([5]).cuda()
    a.requires_grad = True
    prod_val = torch.cumprod(a, dim=0)
    # Gradients for cumprod of ones [1, 1, 1, 1, 1] are [5, 4, 3, 2, 1]
    prod_val.sum().backward()
    expected_grad_1d = torch.tensor([5., 4., 3., 2., 1.], device='cuda:0')
    assert torch.allclose(a.grad, expected_grad_1d), f"Expected {expected_grad_1d}, got {a.grad}"

    # Case 2: 2D tensor, cumprod over specified dimension (dim=0)
    b = torch.ones([2, 5]).cuda()
    b.requires_grad = True
    prod_val = torch.cumprod(b, dim=0)
    # For dim=0 with 2 rows, gradients for the first row are 2, second row are 1.
    prod_val.sum().backward()
    expected_grad_2d = torch.tensor([[2., 2., 2., 2., 2.], [1., 1., 1., 1., 1.]], device='cuda:0')
    assert torch.allclose(b.grad, expected_grad_2d), f"Expected {expected_grad_2d}, got {b.grad}"

if __name__ == "__main__":
    test_torch_cumprod_gradient()
    print("Test passed.")