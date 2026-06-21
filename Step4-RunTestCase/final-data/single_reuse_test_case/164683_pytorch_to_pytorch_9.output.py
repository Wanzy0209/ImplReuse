import torch
import sys

# Configuration based on the bug report
# Use try-except to handle cases where torch._dynamo or torch._inductor are not available
try:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._inductor.config.emulate_precision_casts = True
except AttributeError:
    print("Warning: torch._dynamo or torch._inductor not found. Skipping configuration.")

def foo(arg0, arg1, arg2, sentinel):
    t0 = arg0 # size=(4, 4), stride=(4, 1), dtype=int64, device=cuda
    # Replaced torch.tanh with torch.rand
    # torch.tanh(t0) would typically return a float tensor if t0 is int, 
    # or error depending on version/context. torch.rand generates a float tensor.
    t1 = torch.rand(t0.shape, dtype=torch.float, device=t0.device)
    
    t2 = arg1 # size=(5,), stride=(1,), dtype=int64, device=cuda
    t3 = t2.min() # size=(), stride=(), dtype=int64, device=cuda
    
    # t1 is float, t3.item() is int. fill_ will cast int to float.
    t4 = t1.clone(); t4.fill_(t3.item()) # size=(4, 4), stride=(4, 1), dtype=float, device=cuda
    
    t5 = arg2 # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t6 = torch.nn.functional.relu(t5) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t7 = torch.nn.functional.silu(t6) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    
    # t4 is float, clamp works, to(torch.long) casts to int64 for embedding
    t8 = torch.nn.functional.embedding(torch.clamp(t4, 0, t7.size(0) - 1).to(torch.long), t7) # size=(4, 4, 4), stride=(16, 4, 1), dtype=bfloat16, device=cuda
    t9 = t8.min() # size=(), stride=(), dtype=bfloat16, device=cuda
    output = t9 + sentinel  # output tensor with sentinel for gradient flow
    return output

# Setup arguments
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
    # Check if torch.compile exists to avoid AttributeError in older PyTorch versions
    if hasattr(torch, 'compile'):
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, sentinel)
        out_compiled.sum().backward()
        print('Compile Success! ')
    else:
        print('Skipping Compile Test: torch.compile is not available in this PyTorch version.')