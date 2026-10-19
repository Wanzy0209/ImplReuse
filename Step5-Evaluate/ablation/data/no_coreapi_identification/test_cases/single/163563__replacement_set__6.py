import torch
import sys

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    # Replacement: torch.sigmoid -> torch.div
    t1 = torch.div(t0, 1.0) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    t2 = arg1 # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
    # Replacement: torch.sigmoid -> torch.div
    t3 = torch.div(t2, 1.0) # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
    t4 = arg2 # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda
    # Replacement: torch.exp -> torch.mul
    t5 = torch.mul(t4, 1.0) # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda
    # Replacement: torch.baddbmm -> torch.atleast_3d
    t6 = torch.atleast_3d(t1) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    t7 = t6.reshape((193, 386, 459)) # size=(193, 386, 459), stride=(177174, 459, 1), dtype=bfloat16, device=cuda
    output = t7  # output tensor
    return output

arg0 = torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
arg1 = torch.rand([5699097, 6, 256], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
arg2 = torch.rand([5699097, 256, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')