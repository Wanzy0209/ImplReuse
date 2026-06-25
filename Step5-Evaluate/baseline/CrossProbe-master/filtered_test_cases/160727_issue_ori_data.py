import torch

def test_addmm_activation():
    # Simplified reproduction attempt
    x = torch.randn(10, 20)
    mat1 = torch.randn(10, 30)
    mat2 = torch.randn(30, 20)
    bias = torch.randn(20)
    result = torch.addmm(bias, mat1, mat2)
    return result

test_addmm_activation()