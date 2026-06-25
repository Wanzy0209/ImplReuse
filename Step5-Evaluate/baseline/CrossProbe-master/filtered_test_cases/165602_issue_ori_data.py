import torch
def test_fn_fwgrad_bwgrad_cholesky_solve():
    device = 'cuda'
    dtype = torch.float64
    a = torch.randn(10, 10, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(10, dtype=dtype, device=device, requires_grad=True)
    out = torch.linalg.solve(a, b)
    grad = torch.randn_like(out)
    out.backward(grad)
    return a.grad, b.grad