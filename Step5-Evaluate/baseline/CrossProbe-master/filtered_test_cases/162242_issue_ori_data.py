import torch

def test_scatter_determinism():
    inputs = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4])
    src = torch.arange(-99, -99 + (2 * 2 * 3), dtype=torch.float32).reshape([2, 2, 3])
    index = torch.tensor([[[0, 1, 0], [1, 1, 0]], [[1, 0, 1], [0, 0, 1]]], dtype=torch.int64)
    
    inputs.requires_grad = True
    src.requires_grad = True
    
    res = torch.scatter(inputs, 1, index, src)
    res.backward(torch.ones_like(res))
    
    print('Result:', res)
    print('Input grad:', inputs.grad)
    print('Src grad:', src.grad)

if __name__ == '__main__':
    test_scatter_determinism()