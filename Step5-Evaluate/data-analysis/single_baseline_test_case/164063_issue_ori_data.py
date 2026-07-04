# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import sys
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
    t0 = arg0 # size=(36, 7112, 1, 1), stride=(7112, 1, 1, 1), dtype=bfloat16, device=cuda
    t1 = t0.reshape((28, 24, 3, 127)) # size=(28, 24, 3, 127), stride=(9144, 381, 127, 1), dtype=bfloat16, device=cuda
    t2 = t1.var(dim=2) # size=(28, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
    t3 = arg1 # size=(30, 24), stride=(30, 1), dtype=int64, device=cuda
    t4 = arg2 # size=(512, 127), stride=(512, 1), dtype=bfloat16, device=cuda
    t5 = torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4) # size=(30, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
    t6 = arg3 # size=(30, 24, 15), stride=(720, 24, 1), dtype=bfloat16, device=cuda
    t7 = torch.nn.functional.pad(t6, [0, 1], mode='constant', value=0.0) # size=(30, 24, 16), stride=(384, 16, 1), dtype=bfloat16, device=cuda
    t8 = arg4 # size=(30, 4, 16, 127), stride=(8128, 2032, 127, 1), dtype=bfloat16, device=cuda
    t9 = t8.sum(dim=1) # size=(30, 16, 127), stride=(2032, 127, 1), dtype=bfloat16, device=cuda
    t10 = torch.baddbmm(t5, t7, t9) # size=(30, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
    t11 = torch.cat([t2, t10], dim=0) # size=(58, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
    output = t11 + sentinel  # output tensor with sentinel for gradient flow
    return output

arg0 = torch.rand([36, 7112, 1, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(36, 7112, 1, 1), stride=(7112, 1, 1, 1), dtype=bfloat16, device=cuda
arg1 = torch.randint(0, 512, [30, 24], dtype=torch.int64, device='cuda') # size=(30, 24), stride=(30, 1), dtype=int64, device=cuda
arg2 = torch.rand([512, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(512, 127), stride=(512, 1), dtype=bfloat16, device=cuda
arg3 = torch.rand([30, 24, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(30, 24, 15), stride=(720, 24, 1), dtype=bfloat16, device=cuda
arg4 = torch.rand([30, 4, 16, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(30, 4, 16, 127), stride=(8128, 2032, 127, 1), dtype=bfloat16, device=cuda
sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True) # Sentinel for gradient flow
if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
    out_compiled.sum().backward()
    print('Compile Success! ✅')