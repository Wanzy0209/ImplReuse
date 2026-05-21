import torch
import sys

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0 # size=(5, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    # Adapted: Replace torch.addmm with torch.argmin
    # argmin returns indices (int64), so we cast to float32 to allow subsequent operations (norm, pow)
    t3 = torch.argmin(t0, dim=1).to(torch.float32) 
    
    t4 = t3.norm() # size=(), stride=(), dtype=float32, device=cuda
    t5 = arg3 # size=(3, 4, 5, 2), stride=(40, 10, 2, 1), dtype=float32, device=cuda
    t6 = t5.var(dim=0) # size=(4, 5, 2), stride=(10, 2, 1), dtype=float32, device=cuda
    t7 = t6.var() # size=(), stride=(), dtype=float32, device=cuda
    t8 = arg4 # size=(), stride=(), dtype=float32, device=cuda
    t9 = torch.nn.functional.relu(t8) # size=(), stride=(), dtype=float32, device=cuda
    t10 = t7 + t4 + t9 # size=(), stride=(), dtype=float32, device=cuda
    t11 = torch.pow(torch.pow(t4, t7), t10) # size=(), stride=(), dtype=float32, device=cuda
    output = t11  # output tensor
    return output

arg0 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(5, 4), stride=(4, 1), dtype=bfloat16, device=cuda
arg1 = torch.rand([5, 1024], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(5, 1024), stride=(1024, 1), dtype=bfloat16, device=cuda
arg2 = torch.rand([1024, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(1024, 4), stride=(4, 1), dtype=bfloat16, device=cuda
arg3 = torch.rand([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True) # size=(3, 4, 5, 2), stride=(40, 10, 2, 1), dtype=float32, device=cuda
arg4 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True) # size=(), stride=(), dtype=float32, device=cuda

if __name__ == '__main__':
    # Note: Backward pass is omitted because torch.argmin is non-differentiable,
    # which would cause a runtime error in the original test structure.
    
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    print('Eager Success! ')
    
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    print('Compile Success! ')
    
    # Compare outputs (forward)
    # Since argmin returns discrete indices, we check for exact equality rather than relative difference.
    if not torch.equal(out_eager, out_compiled):
        print(f' Outputs differ!')
        print('out_eager:', out_eager)
        print('out_compiled:', out_compiled)
        sys.exit(1)
    else:
        print('Outputs match exactly! ')