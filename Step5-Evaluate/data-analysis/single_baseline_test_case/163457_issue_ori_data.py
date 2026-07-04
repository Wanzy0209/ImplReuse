# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import sys
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(1,), stride=(1,), dtype=int64, device=cuda
    t2 = arg2 # size=(395,), stride=(1,), dtype=bfloat16, device=cuda
    t3 = torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2) # size=(3, 395, 202, 357), stride=(28485030, 72114, 357, 1), dtype=bfloat16, device=cuda
    t4 = t3.max(dim=0).values # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda
    t5 = torch.sigmoid(t4) # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda
    output = t5  # output tensor
    return output

arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda') # size=(1,), stride=(1,), dtype=int64, device=cuda
arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(395,), stride=(1,), dtype=bfloat16, device=cuda
if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ✅')
    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)