import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Apply the configuration from the bug report
# Check if _inductor exists to prevent AttributeError in environments without it
if hasattr(torch, '_inductor'):
    torch._inductor.config.combo_kernels = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU tensors to ensure the test is runnable without specific GPU setups
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Adapted function using the similar API (torch.distributed.recv_object_list)
# Check if torch.compile is available to support older PyTorch versions
if hasattr(torch, 'compile'):
    @torch.compile
    def fn_recv(obj_list):
        dist.recv_object_list(obj_list)
else:
    def fn_recv(obj_list):
        dist.recv_object_list(obj_list)

def run(rank, world_size):
    setup(rank, world_size)
    if rank == 0:
        # Sender
        data = [torch.rand(10)]
        dist.send_object_list(data, dst=1)
    else:
        # Receiver
        obj_list = [None]
        # Call the compiled function containing the similar API
        fn_recv(obj_list)
        # Verify reception
        assert obj_list[0] is not None
        assert isinstance(obj_list[0], torch.Tensor)
        print(f"Rank {rank} received object successfully.")
    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)