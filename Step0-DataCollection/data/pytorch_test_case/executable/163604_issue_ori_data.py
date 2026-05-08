import torch
import sys
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1):
    t0 = arg0 # size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1), dtype=float32, device=cuda
    t1 = t0.mean(dim=0) # size=(503, 64, 504), stride=(32192, 64, 1), dtype=float32, device=cuda
    t2 = torch.nn.functional.relu(t1) # size=(503, 64, 504), stride=(32192, 64, 1), dtype=float32, device=cuda
    t3 = arg1 # size=(5, 16, 1, 64), stride=(1024, 64, 64, 1), dtype=float32, device=cuda
    t4 = t3.sum(dim=0) # size=(16, 1, 64), stride=(1024, 1, 64), dtype=float32, device=cuda
    t5 = t4.transpose(2, 1) # size=(16, 64, 1), stride=(1024, 64, 1), dtype=float32, device=cuda
    t6 = torch.nn.functional.conv1d(t2, t5, stride=1, padding=0) # size=(503, 16, 504), stride=(8064, 504, 1), dtype=float32, device=cuda
    output = t6  # output tensor
    return output

arg0 = torch.rand([4, 503, 64, 504], dtype=torch.float32, device='cuda', requires_grad=True) # size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1), dtype=float32, device=cuda
arg1 = torch.rand([5, 16, 1, 64], dtype=torch.float32, device='cuda', requires_grad=True) # size=(5, 16, 1, 64), stride=(1024, 64, 64, 1), dtype=float32, device=cuda
if __name__ == '__main__':
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ✅')