import torch
import sys

# Replicate the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1):
    # t0, t1, t2 logic remains the same to preserve the context (strides, dtypes)
    t0 = arg0 # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    t1 = t0.max(dim=0).values # size=(261, 17, 358), stride=(358, 93438, 1), dtype=bfloat16, device=cuda
    t2 = t1.transpose(1, 0) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=bfloat16, device=cuda
    
    # Adaptation for torch.nn.functional.adaptive_avg_pool2d
    # The original conv1d transformed (17, 64, 358) to (17, 261, 358).
    # AdaptiveAvgPool2d preserves channels, so we adjust arg1 to have 261 channels to match t2.
    t3 = arg1 # size=(17, 261, 358), dtype=float32, device=cuda
    t4 = torch.exp(t3) # size=(17, 261, 358), dtype=float32, device=cuda
    
    # adaptive_avg_pool2d expects 4D input (N, C, H, W). 
    # We reshape t4 from (17, 261, 358) to (17, 261, 358, 1).
    t4_4d = t4.unsqueeze(-1)
    
    # Apply adaptive_avg_pool2d. We use output_size=(358, 1) to maintain spatial dimensions
    # compatible with the subsequent operations, effectively testing the API within the graph.
    t7_4d = torch.nn.functional.adaptive_avg_pool2d(t4_4d, output_size=(358, 1))
    
    # Restore shape to (17, 261, 358) to match t2
    t7 = t7_4d.squeeze(-1) # size=(17, 261, 358), dtype=float32, device=cuda
    
    t8 = t7.clone(); t8.zero_() # size=(17, 261, 358), dtype=float32, device=cuda
    t9 = t2 * t7 * t8 # size=(17, 261, 358), dtype=float32, device=cuda
    output = t9  # output tensor
    return output

# Adjusted inputs
# arg0 remains the same
arg0 = torch.rand([2, 261, 17, 358], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# arg1 adjusted: channels changed from 64 to 261 to match t2's shape after pooling
arg1 = torch.rand([17, 261, 358], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    print('Compile Success! ')