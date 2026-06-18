import torch
import sys
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0):
    t0 = arg0 # size=(), stride=(), dtype=bfloat16, device=cuda
    t1 = torch.softmax(t0, dim=0) # size=(), stride=(), dtype=bfloat16, device=cuda
    t2 = torch.nn.functional.gelu(t1) # size=(), stride=(), dtype=bfloat16, device=cuda
    t3 = torch.softmax(t2, dim=0) # size=(), stride=(), dtype=bfloat16, device=cuda
    output = t3  # output tensor
    return output

arg0 = torch.rand([], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(), stride=(), dtype=bfloat16, device=cuda
if __name__ == '__main__':
    out_eager = foo(arg0)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0)
    out_compiled.sum().backward()
    print('Compile Success! ✅')