# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0, arg1):
    t0 = arg0 # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
    t1 = arg1 # size=(), stride=(), dtype=float32, device=cuda
    t2 = t0.clone(); t2.fill_diagonal_(t1.item()) # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
    return t2

arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True) # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True) # size=(), stride=(), dtype=float32, device=cuda
if __name__ == '__main__':
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ✅')