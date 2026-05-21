import torch

# Replicate the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = torch.sigmoid(t0)
    t2 = arg1
    t3 = torch.sigmoid(t2)
    t4 = arg2
    t5 = torch.exp(t4)
    t6 = torch.baddbmm(t1, t3, t5)
    t7 = t6.reshape((193, 386, 459))
    
    # Adaptation: Use torch.all to verify the behavior on the large tensor.
    # We check if all elements are greater than a threshold.
    output = torch.all(t7 > -1.0)
    return output

# Generate inputs matching the original bug report's dimensions and types
# Note: requires_grad is removed as torch.all returns a boolean scalar, 
# which typically doesn't support backward in the same way.
arg0 = torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cuda')
arg1 = torch.rand([5699097, 6, 256], dtype=torch.bfloat16, device='cuda')
arg2 = torch.rand([5699097, 256, 1], dtype=torch.bfloat16, device='cuda')

if __name__ == '__main__':
    # Test Eager mode
    try:
        out_eager = foo(arg0, arg1, arg2)
        print(f'Eager Success!  Result: {out_eager}')
    except RuntimeError as e:
        print(f'Eager Failed!  Error: {e}')
        sys.exit(1)

    # Test Compiled mode
    try:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        print(f'Compile Success!  Result: {out_compiled}')
    except RuntimeError as e:
        print(f'Compile Failed!  Error: {e}')
        sys.exit(1)

    # Verify consistency
    assert out_eager == out_compiled, f"Divergence detected! Eager: {out_eager}, Compiled: {out_compiled}"
    print('Test Passed! ')