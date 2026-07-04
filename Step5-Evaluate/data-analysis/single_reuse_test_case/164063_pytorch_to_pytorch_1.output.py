import torch

# Reproduce the specific configuration that triggers the bug
# Guard against older PyTorch versions where _dynamo does not exist
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
    t0 = arg0
    t1 = t0.reshape((28, 24, 3, 127))
    # Replaced torch.var with torch.std
    t2 = t1.std(dim=2)
    t3 = arg1
    t4 = arg2
    t5 = torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4)
    t6 = arg3
    t7 = torch.nn.functional.pad(t6, [0, 1], mode='constant', value=0.0)
    t8 = arg4
    t9 = t8.sum(dim=1)
    t10 = torch.baddbmm(t5, t7, t9)
    t11 = torch.cat([t2, t10], dim=0)
    output = t11 + sentinel
    return output

# Setup inputs
arg0 = torch.rand([36, 7112, 1, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.randint(0, 512, [30, 24], dtype=torch.int64, device='cuda')
arg2 = torch.rand([512, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg3 = torch.rand([30, 24, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg4 = torch.rand([30, 4, 16, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Test Eager mode
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Test Compiled mode
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        out_compiled.sum().backward()
        print('Compile Success! ')

        # Verify results match
        assert torch.allclose(out_eager, out_compiled), "Divergence between eager and compiled outputs"
    else:
        print("torch.compile is not available in this environment. Skipping compiled test.")