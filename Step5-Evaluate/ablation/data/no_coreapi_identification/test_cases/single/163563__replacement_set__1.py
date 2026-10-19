import torch
import torch.distributed as dist
import os

# Initialize distributed environment for torch.distributed.all_reduce
# Using a simple TCP store for single-process testing
if not dist.is_initialized():
    try:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://localhost:29500",
            rank=0,
            world_size=1
        )
    except Exception as e:
        print(f"Skipping distributed init: {e}")

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0
    # Replacement: torch.sigmoid -> torch.tanh
    t1 = torch.tanh(t0)
    t2 = arg1
    # Replacement: torch.sigmoid -> torch.tanh
    t3 = torch.tanh(t2)
    t4 = arg2
    # Replacement: torch.exp -> torch.log
    t5 = torch.log(t4)
    
    # Replacement: torch.baddbmm -> torch.distributed.all_reduce
    # baddbmm(t1, t3, t5) is replaced by all_reduce on t1.
    # We clone t1 to ensure we have a tensor to assign to t6, as all_reduce is in-place.
    t6 = t1.clone()
    dist.all_reduce(t6)
    
    t7 = t6.reshape((193, 386, 459))
    output = t7
    return output

# Replacement: torch.rand -> torch.flipud
# We use torch.ones as the base tensor to flip, ensuring torch.rand is replaced.
arg0 = torch.flipud(torch.ones([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True))
arg1 = torch.flipud(torch.ones([5699097, 6, 256], dtype=torch.bfloat16, device='cuda', requires_grad=True))
arg2 = torch.flipud(torch.ones([5699097, 256, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True))

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')