import torch
import torch.distributed as dist
import os
import sys

# Setup distributed environment for torch.distributed.all_gather
def setup():
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        # Use gloo backend for general compatibility, though nccl is preferred for CUDA
        backend = 'gloo' 
        try:
            dist.init_process_group(backend=backend, rank=0, world_size=1)
        except Exception as e:
            print(f"Distributed setup failed: {e}")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

setup()

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0
    # Replacement: torch.sigmoid -> torch.exp
    t1 = torch.exp(t0)
    t2 = arg1
    # Replacement: torch.sigmoid -> torch.exp
    t3 = torch.exp(t2)
    t4 = arg2
    # Replacement: torch.exp -> torch.tanh
    t5 = torch.tanh(t4)
    
    # Replacement: torch.baddbmm -> torch.distributed.all_gather
    # Original: t6 = torch.baddbmm(t1, t3, t5)
    # To maintain runnable logic with a collective op, we gather t3 and stack results.
    # Note: all_gather modifies the list in-place.
    if dist.is_initialized():
        gather_list = [torch.zeros_like(t3) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, t3)
        # Stack results to mimic a tensor output for the subsequent reshape
        t6 = torch.stack(gather_list).sum(0)
    else:
        # Fallback if distributed failed to init, though setup() should handle it
        t6 = t3 

    t7 = t6.reshape((193, 386, 459))
    output = t7
    return output

# Replacement: torch.rand -> torch.fliplr
# torch.fliplr requires an input tensor, so we generate a base tensor using torch.randn
# (which is not in the replacement list) and then apply fliplr.
base0 = torch.randn([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = torch.fliplr(base0)

base1 = torch.randn([5699097, 6, 256], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.fliplr(base1)

base2 = torch.randn([5699097, 256, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.fliplr(base2)

if __name__ == '__main__':
    try:
        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print('Eager Success! ')
        
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print('Compile Success! ')
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()