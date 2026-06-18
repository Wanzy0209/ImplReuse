import torch

def test_index_put_with_unique_indices():
    torch.manual_seed(2025)

    def foo(x):
        # Leverage torch.unique to generate the target indices [2, 3]
        # This reuses the similar API while preserving the original assignment logic
        target_indices = torch.unique(torch.tensor([2, 3]))
        
        x[0].sin_()
        x[1].sin_()
        y = torch.zeros_like(x)
        
        # Perform the assignment using the unique indices
        y[target_indices[0]] = x[0]
        y[target_indices[1]] = x[1]
        return y

    cfoo = torch.compile(foo)
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
    cx = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
    
    res = foo(x)
    cres = cfoo(cx)
    
    # The bug causes cres[2] to be sin(sin(x[0])) instead of sin(x[0])
    # This assertion should fail if the bug is present
    torch.testing.assert_close(res, cres)

if __name__ == "__main__":
    test_index_put_with_unique_indices()