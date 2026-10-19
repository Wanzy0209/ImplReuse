import torch
import torch.distributed as dist
import os

# Initialize distributed environment for torch.distributed.reduce_scatter
# Using a single process to keep the test minimal and runnable
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

if not dist.is_initialized():
    # Use 'gloo' backend as it is generally available and works with CUDA tensors
    # For a single-process test, world_size=1
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    
    # Replacement: torch.sigmoid -> torch.log
    t1 = torch.log(t0) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    
    t2 = arg1 # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
    
    # Replacement: torch.sigmoid -> torch.log
    t3 = torch.log(t2) # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
    
    t4 = arg2 # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda
    
    # Replacement: torch.exp -> torch.square
    t5 = torch.square(t4) # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda
    
    # Replacement: torch.baddbmm -> torch.distributed.reduce_scatter
    # baddbmm(t1, t3, t5) output shape is (5699097, 6, 1)
    # reduce_scatter requires an output tensor and a list of input tensors.
    # To maintain the output shape (5699097, 6, 1) for the subsequent reshape,
    # we reshape inputs to match and use t1 as the output buffer base.
    t6 = torch.empty_like(t1) # size=(5699097, 6, 1)
    
    # Reshape t3 and t5 to be compatible with reduce_scatter list requirements
    # and the output shape t6. Since world_size=1, output shape = input shape.
    # We slice/expand to (5699097, 6, 1).
    t3_view = t3[..., :1] # (5699097, 6, 1)
    t5_view = t5[:, :6, :] # (5699097, 6, 1)
    
    dist.reduce_scatter(t6, [t3_view, t5_view])
    
    t7 = t6.reshape((193, 386, 459)) # size=(193, 386, 459), stride=(177174, 459, 1), dtype=bfloat16, device=cuda
    output = t7  # output tensor
    return output

# Replacement: torch.rand -> torch.tanh
# torch.tanh requires an input tensor. We use torch.randn to generate the input.
arg0 = torch.tanh(torch.randn([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True))
arg1 = torch.tanh(torch.randn([5699097, 6, 256], dtype=torch.bfloat16, device='cuda', requires_grad=True))
arg2 = torch.tanh(torch.randn([5699097, 256, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True))

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')