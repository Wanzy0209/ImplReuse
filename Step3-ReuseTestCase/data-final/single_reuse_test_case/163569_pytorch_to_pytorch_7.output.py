import torch
import sys

# Replicate the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1):
    # Adapted shapes for UpsamplingBilinear2d (4D input required)
    # arg0: size=(2, 64, 17, 20), dtype=bfloat16, device=cuda
    # arg1: size=(17, 64, 10, 10), dtype=float32, device=cuda
    
    t0 = arg0
    t1 = t0.max(dim=0).values # size=(64, 17, 20)
    t2 = t1.transpose(1, 0)  # size=(17, 64, 20)
    
    # Reshape t2 to be broadcastable with the 4D output of the upsampling operation
    t2 = t2.unsqueeze(-1)    # size=(17, 64, 20, 1)
    
    t3 = arg1
    t4 = torch.exp(t3)       # size=(17, 64, 10, 10)
    
    # Original call: torch.nn.functional.conv1d(t4, t6, stride=1, padding=0)
    # Adapted call: torch.nn.functional.upsample_bilinear (functional equivalent of torch.nn.UpsamplingBilinear2d)
    # UpsamplingBilinear2d requires 4D input (N, C, H, W).
    # We use scale_factor=2 to change spatial dimensions from (10, 10) to (20, 20).
    t7 = torch.nn.functional.upsample_bilinear(t4, scale_factor=2, align_corners=True) 
    # size=(17, 64, 20, 20), dtype=float32, device=cuda
    
    t8 = t7.clone()
    t8.zero_()               # size=(17, 64, 20, 20)
    
    t9 = t2 * t7 * t8        # size=(17, 64, 20, 20)
    output = t9
    return output

if __name__ == '__main__':
    # Adapted inputs to match the new tensor shapes required by UpsamplingBilinear2d
    arg0 = torch.rand([2, 64, 17, 20], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg1 = torch.rand([17, 64, 10, 10], dtype=torch.float32, device='cuda', requires_grad=True)
    
    # Eager execution
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ')