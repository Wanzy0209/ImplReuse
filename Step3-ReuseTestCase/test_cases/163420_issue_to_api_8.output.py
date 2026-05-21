import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0, arg1):
    # arg0: size=(2, 2), stride=(2, 1), dtype=float32, device=cuda
    # arg1: size=(), stride=(), dtype=int64, device=cuda (0-d tensor)
    
    # Original bug used fill_diagonal_ with a 0-d value.
    # This test uses index_select with a 0-d index.
    # The torch._refs.index_select implementation explicitly handles index.ndim == 0
    # by calling unsqueeze(0).
    
    # We select along dimension 0 using the 0-d index tensor.
    return torch.index_select(arg0, 0, arg1)

# Setup inputs
# arg0 is a 2x2 matrix
arg0 = torch.empty([2, 2], dtype=torch.float32, device='cuda', requires_grad=True)
# arg1 is a 0-d tensor (scalar) used as an index. 
# Note: index_select requires integer indices.
arg1 = torch.tensor(0, dtype=torch.int64, device='cuda')

if __name__ == '__main__':
    # Test Eager mode
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Test Compiled mode
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ')