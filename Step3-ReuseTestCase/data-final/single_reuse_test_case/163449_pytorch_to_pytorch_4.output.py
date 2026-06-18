import torch
import sys

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4):
    # Adapted to use torch.argmax instead of torch.addmm
    # arg1 is (5, 1024), argmax along dim 1 results in (5,)
    t3 = torch.argmax(arg1, dim=1)

    t4 = t3.norm() # size=(), stride=(), dtype=float32 (from int64 norm)
    t5 = arg3
    t6 = t5.var(dim=0)
    t7 = t6.var()
    t8 = arg4
    t9 = torch.nn.functional.relu(t8)
    t10 = t7 + t4 + t9
    t11 = torch.pow(torch.pow(t4, t7), t10)
    output = t11
    return output

arg0 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.rand([5, 1024], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.rand([1024, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg3 = torch.rand([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)
arg4 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    out_compiled.sum().backward()
    print('Compile Success! ')
    
    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)