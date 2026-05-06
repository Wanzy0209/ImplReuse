import torch
import sys
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel):
    t0 = arg0 # size=(93, 62, 23), stride=(1426, 23, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(93, 62, 11), stride=(682, 11, 1), dtype=bfloat16, device=cuda
    t2 = arg2 # size=(93, 62, 10), stride=(620, 10, 1), dtype=bfloat16, device=cuda
    t3 = arg3 # size=(93, 62, 81), stride=(5022, 81, 1), dtype=bfloat16, device=cuda
    t4 = arg4 # size=(93, 62, 2), stride=(124, 2, 1), dtype=bfloat16, device=cuda
    t5 = torch.cat([t0, t1, t2, t3, t4], dim=2) # size=(93, 62, 127), stride=(23622, 635, 4), dtype=bfloat16, device=cuda
    t6 = t5.contiguous() # size=(93, 62, 127), stride=(7874, 127, 1), dtype=bfloat16, device=cuda
    t7 = arg5 # size=(93, 62, 8), stride=(5766, 62, 1), dtype=bfloat16, device=cuda
    t8 = torch.exp(t7) # size=(93, 62, 8), stride=(5766, 62, 1), dtype=bfloat16, device=cuda
    t9 = torch.rms_norm(t8, (62, 8)) # size=(93, 62, 8), stride=(496, 8, 1), dtype=bfloat16, device=cuda
    t10 = arg6 # size=(77, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
    t11 = torch.exp(t10) # size=(77, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
    t12 = arg7 # size=(16, 8, 15), stride=(128, 8, 1), dtype=bfloat16, device=cuda
    t13 = torch.nn.functional.interpolate(t12, size=(127,), mode='nearest') # size=(16, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
    t14 = torch.cat([t11, t13], dim=0) # size=(93, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
    t15 = torch.baddbmm(t6, t9, t14) # size=(93, 62, 127), stride=(7874, 127, 1), dtype=bfloat16, device=cuda
    output = t15 + sentinel  # output tensor with sentinel for gradient flow
    return output

arg0 = torch.rand([93, 62, 23], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(93, 62, 23), stride=(1426, 23, 1), dtype=bfloat16, device=cuda
arg1 = torch.rand([93, 62, 11], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(93, 62, 11), stride=(682, 11, 1), dtype=bfloat16, device=cuda
arg2 = torch.rand([93, 62, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(93, 62, 10), stride=(620, 10, 1), dtype=bfloat16, device=cuda
arg3 = torch.rand([93, 62, 81], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(93, 62, 81), stride=(5022, 81, 1), dtype=bfloat16, device=cuda
arg4 = torch.rand([93, 62, 2], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(93, 62, 2), stride=(124, 2, 1), dtype=bfloat16, device=cuda
arg5 = torch.rand([93, 62, 8], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(93, 62, 8), stride=(5766, 62, 1), dtype=bfloat16, device=cuda
arg6 = torch.rand([77, 8, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(77, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
arg7 = torch.rand([16, 8, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(16, 8, 15), stride=(128, 8, 1), dtype=bfloat16, device=cuda
sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True) # Sentinel for gradient flow
if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
    out_compiled.sum().backward()
    print('Compile Success! ✅')