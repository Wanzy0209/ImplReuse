import torch
import sys

# Check for PyTorch 2.0+ features to handle missing dependencies gracefully
HAS_DYNAMO = hasattr(torch, '_dynamo')
HAS_INDUCTOR = hasattr(torch, '_inductor')
HAS_COMPILE = hasattr(torch, 'compile')

if HAS_DYNAMO:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

if HAS_INDUCTOR:
    torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, sentinel):
    t0 = arg0 # size=(4, 4), stride=(4, 1), dtype=int64, device=cuda
    # Replaced torch.tanh with torch.exp
    t1 = torch.exp(t0) # size=(4, 4), stride=(4, 1), dtype=float32 (promoted from int64), device=cuda
    t2 = arg1 # size=(5,), stride=(1,), dtype=int64, device=cuda
    t3 = t2.min() # size=(), stride=(), dtype=int64, device=cuda
    t4 = t1.clone(); t4.fill_(t3.item()) # size=(4, 4), stride=(4, 1), dtype=float32, device=cuda
    t5 = arg2 # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t6 = torch.nn.functional.relu(t5) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t7 = torch.nn.functional.silu(t6) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t8 = torch.nn.functional.embedding(torch.clamp(t4, 0, t7.size(0) - 1).to(torch.long), t7) # size=(4, 4, 4), stride=(16, 4, 1), dtype=bfloat16, device=cuda
    t9 = t8.min() # size=(), stride=(), dtype=bfloat16, device=cuda
    output = t9 + sentinel  # output tensor with sentinel for gradient flow
    return output

arg0 = torch.randint(0, 1000, [4, 4], dtype=torch.int64, device='cuda') # size=(4, 4), stride=(4, 1), dtype=int64, device=cuda
arg1 = torch.randint(0, 1000, [5], dtype=torch.int64, device='cuda') # size=(5,), stride=(1,), dtype=int64, device=cuda
arg2 = torch.rand([5000, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True) # Sentinel for gradient flow

if __name__ == '__main__':
    # Test Eager Mode
    out_eager = foo(arg0, arg1, arg2, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Test Compiled Mode
    if HAS_COMPILE:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, sentinel)
        out_compiled.sum().backward()
        print('Compile Success! ')
    else:
        print('Compile Skipped: torch.compile not available (requires PyTorch 2.0+)')