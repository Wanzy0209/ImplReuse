import torch
import sys

# Reproduce the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1):
    # arg0: bfloat16, arg1: float32
    # t0 = arg0 # size=(2, 1, 3, 4, 5, 6), dtype=bfloat16, device=cuda
    t0 = arg0
    # t1 = t0.max(dim=0).values # size=(1, 3, 4, 5, 6), dtype=bfloat16, device=cuda
    t1 = t0.max(dim=0).values
    # t2 = t1.transpose(0, 1) # size=(3, 1, 4, 5, 6), dtype=bfloat16, device=cuda
    # Note: Original used transpose(1, 0) on 3D tensor. Here we use transpose(0, 1) on 4D tensor
    # to maintain the non-contiguous nature and shape manipulation logic.
    t2 = t1.transpose(0, 1)
    
    # t3 = arg1 # size=(3, 1, 5, 6, 7), dtype=float32, device=cuda
    t3 = arg1
    # t4 = torch.exp(t3) # size=(3, 1, 5, 6, 7), dtype=float32, device=cuda
    t4 = torch.exp(t3)
    
    # t7 = torch.nn.functional.avg_pool3d(t4, kernel_size=2, stride=1)
    # Input size=(3, 1, 5, 6, 7), kernel=2, stride=1 -> Output size=(3, 1, 4, 5, 6)
    # This replaces the conv1d call from the original issue.
    t7 = torch.nn.functional.avg_pool3d(t4, kernel_size=2, stride=1)
    
    # t8 = t7.clone(); t8.zero_()
    t8 = t7.clone()
    t8.zero_()
    
    # t9 = t2 * t7 * t8
    # t2 size=(3, 1, 4, 5, 6), t7 size=(3, 1, 4, 5, 6), t8 size=(3, 1, 4, 5, 6)
    t9 = t2 * t7 * t8
    
    output = t9
    return output

# Generate inputs matching the adapted shapes
# arg0: (2, 1, 3, 4, 5, 6) bfloat16
arg0 = torch.rand([2, 1, 3, 4, 5, 6], dtype=torch.bfloat16, device='cuda', requires_grad=True)

# arg1: (3, 1, 5, 6, 7) float32
arg1 = torch.rand([3, 1, 5, 6, 7], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Test Eager mode
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Test Compiled mode
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ')