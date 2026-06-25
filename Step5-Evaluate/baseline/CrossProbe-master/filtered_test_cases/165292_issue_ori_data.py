import torch
if torch.cuda.is_available() and torch.version.hip:
    device = torch.device('cuda:0')
    input_tensor = torch.randn(2, 5, 5, dtype=torch.float64, device=device)
    input_tensor = torch.matmul(input_tensor, input_tensor.transpose(-2, -1))
    expected = torch.linalg.cholesky_inverse(torch.linalg.cholesky(input_tensor))
    result = torch.linalg.cholesky_inverse(torch.linalg.cholesky(input_tensor))
    torch.testing.assert_close(expected, result, rtol=1e-7, atol=1e-7)