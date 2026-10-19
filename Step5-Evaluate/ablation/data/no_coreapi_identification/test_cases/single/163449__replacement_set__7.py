import tensorflow as tf
import torch
import sys

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0 # size=(5, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(5, 1024), stride=(1024, 1), dtype=bfloat16, device=cuda
    t2 = arg2 # size=(1024, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    
    # Replacement: torch.addmm -> torch.quantile (Similar API)
    # torch.quantile requires float32 or float64, so we cast t1 from bfloat16
    # We use a fixed quantile value (0.5) for the test.
    t1_float = t1.to(torch.float32)
    t3 = torch.quantile(t1_float, 0.5, dim=0) # size=(1024,), stride=(1,), dtype=float32, device=cuda
    
    t4 = t3.norm() # size=(), stride=(), dtype=float32, device=cuda
    
    t5 = arg3 # size=(3, 4, 5, 2), stride=(40, 10, 2, 1), dtype=float32, device=cuda
    t6 = t5.var(dim=0) # size=(4, 5, 2), stride=(10, 2, 1), dtype=float32, device=cuda
    t7 = t6.var() # size=(), stride=(), dtype=float32, device=cuda
    
    t8 = arg4 # size=(), stride=(), dtype=float32, device=cuda
    
    # Replacement: torch.nn.functional.relu -> torch.nn.functional.hardsigmoid
    t9 = torch.nn.functional.hardsigmoid(t8) # size=(), stride=(), dtype=float32, device=cuda
    
    t10 = t7 + t4 + t9 # size=(), stride=(), dtype=float32, device=cuda
    
    # Note: torch.pow is kept as is. Replacing it with tf.compat.v1.nn.dropout 
    # would introduce stochasticity and cross-framework complexity, breaking 
    # the deterministic numerical comparison logic of the test.
    t11 = torch.pow(torch.pow(t4, t7), t10) # size=(), stride=(), dtype=float32, device=cuda
    
    output = t11  # output tensor
    return output

# Input generation
# Note: torch.rand is kept as is. Replacing it with tf.keras.random.dropout 
# is not feasible for tensor initialization in this context.
arg0 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.rand([5, 1024], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.rand([1024, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg3 = torch.rand([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)
arg4 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Compiled execution
    # Note: torch.compile is kept as is. Replacing it with tf.keras.random.dropout 
    # would prevent the compilation test.
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    out_compiled.sum().backward()
    print('Compile Success! ')
    
    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)