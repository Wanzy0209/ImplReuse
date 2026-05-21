import torch
import sys

# Reproduce the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # arg0: Tensor used to maintain differentiable graph for the test harness
    # arg1: new_state (Tensor) for set_rng_state
    # arg2: device (int/str/device) for set_rng_state
    
    # Call the similar API: torch.cuda.random.set_rng_state
    # This replaces the original torch.nn.functional.group_norm call
    torch.cuda.random.set_rng_state(arg1, arg2)
    
    # Return arg0 to allow the test harness to perform backward pass and comparison
    # Since set_rng_state is a side-effect operation returning None, we pass through 
    # a tensor to verify the compilation context handles the mix of ops correctly.
    return arg0

# Setup inputs
# arg0: A tensor with dynamic-like properties (requires_grad) to test graph compilation
arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)

# arg1: A valid RNG state tensor (ByteTensor)
arg1 = torch.cuda.random.get_rng_state()

# arg2: Device index (int)
arg2 = 0

if __name__ == '__main__':
    # Test Eager execution
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Test Compiled execution
    # Using fullgraph=True and dynamic=True as in the original bug report
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Compare outputs (forward)
    # Since foo returns arg0 directly, outputs should be identical
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    
    # Use a small tolerance for floating point comparison
    if diff > 1e-6:
        print(f' Forward output sums differ significantly!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        sys.exit(1)
        
    print('Test Passed! ')