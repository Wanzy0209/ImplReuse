import torch

# Keeping the configuration from the original bug report to maintain a similar testing environment
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0):
    # Replacing the original fill_diagonal_ call with torch.any
    # Original: t2 = t0.clone(); t2.fill_diagonal_(t1.item())
    # Adapted: Use torch.any on the input tensor
    return torch.any(arg0)

# Initialize inputs on CUDA as per the original bug report
# Using randn to ensure deterministic behavior for the 'any' check
arg0 = torch.randn([1, 1], dtype=torch.float32, device='cuda')

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0)
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0)
    
    # Verify that the compiled output matches the eager output
    assert torch.equal(out_eager, out_compiled), f"Eager and Compiled outputs differ: {out_eager} vs {out_compiled}"
    print('Compile Success! ')