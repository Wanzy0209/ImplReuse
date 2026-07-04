import torch

# Mock torch.compile if not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    torch.compile = lambda f: f

def foo(x):
    x.tan_()
    x = x.t()
    return x.isreal()

if __name__ == "__main__":
    torch.manual_seed(0)
    x1 = torch.randn(4, 6)
    x2 = x1.clone()
    
    # Eager execution
    out1 = foo(x1)
    
    # Compiled execution (inductor)
    cf = torch.compile(foo)
    out2 = cf(x2)
    
    # Verify correctness
    torch.testing.assert_close(out1, out2)