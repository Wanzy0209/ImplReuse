import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Configs from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    if rank == 0:
        # Sender: Create and send a tensor
        # The size of this tensor is unknown to the receiver's graph at compile time
        tensor_to_send = torch.randn(3, 4)
        dist.send_object_list([tensor_to_send], dst=1)
    else:
        # Receiver: Define function using the similar API
        def f_recv():
            obj_list = [None]
            # Similar API: torch.distributed.recv_object_list
            # This receives a tensor with a size that is effectively 'unbacked' 
            # relative to the compilation graph of the receiver.
            dist.recv_object_list(obj_list, src=0)
            
            received_tensor = obj_list[0]
            
            # The problematic operation from the original bug: 
            # Slicing a tensor with a dynamic/unbacked size
            return received_tensor[:-1]

        # Compile the function containing the similar API
        # This mirrors the original test case: torch.compile(f, fullgraph=True)(...)
        try:
            out = torch.compile(f_recv, fullgraph=True)()
            print(f"Rank {rank}: Output shape {out.shape}")
        except Exception as e:
            print(f"Rank {rank}: Error during compiled execution - {e}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Run the test in a multiprocess environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)