import torch

# Configuration to match the original bug report context
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0, arg1):
    """
    Reproduces the original bug logic (fill_diagonal_) while leveraging 
    the pattern from the similar API (tf.compat.v1.nn.dropout).
    
    The TF implementation explicitly converts scalars to tensors:
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
    
    We adapt this pattern to the PyTorch bug scenario by converting the 
    scalar value (extracted via .item()) back into a tensor before 
    passing it to fill_diagonal_.
    """
    t0 = arg0 # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
    t1 = arg1 # size=(), stride=(), dtype=float32, device=cuda
    
    # Original bug logic: clone and fill diagonal
    t2 = t0.clone()
    
    # Leveraging the similar API's pattern (ops.convert_to_tensor):
    # Explicitly convert the scalar value to a tensor with the target dtype.
    # This mimics the TF dropout logic: scale = ops.convert_to_tensor(scale, dtype=x_dtype)
    fill_value = torch.tensor(t1.item(), dtype=t0.dtype)
    
    # Apply the operation using the tensor-converted value
    t2.fill_diagonal_(fill_value)
    
    return t2

# Setup inputs matching the original bug report
arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Test Eager mode
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Test Compiled mode (torch.compile)
    # This checks if the pattern derived from the similar API (TF) 
    # affects the compilation failure observed in the original issue.
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ')