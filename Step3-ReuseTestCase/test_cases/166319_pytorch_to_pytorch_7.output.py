import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    var_node_4 = arg_0
    var_node_3 = torch.squeeze(var_node_4)
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_1 = torch.squeeze(var_node_2)
    var_node_7 = arg_1
    var_node_8 = arg_2
    _input_size_var_node_6 = var_node_7.size(0)
    _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
    var_node_0 = torch.mul(var_node_1, var_node_5)
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

def run_test(rank, world_size):
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use nccl for CUDA devices, gloo for CPU. Original bug uses CUDA.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        torch.manual_seed(1215252001)
        
        # Setup inputs similar to the original bug report
        # Note: Adjusting device to current rank for distributed execution
        arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1)).to(device)
        arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1)).to(device)
        arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1)).to(device)
        sentinel = torch.tensor(1.0, requires_grad=True).to(device)

        # Define a wrapper that executes the logic and sends the result
        # This adapts the original call site (torch.compile) to verify the similar API
        def send_wrapper(a0, a1, a2, s):
            res = fuzzed_program(a0, a1, a2, s)
            # Call the similar API
            dist.send_object_list([res], dst=1)
            return res

        # Compile the wrapper to test for LoweringException
        compiled_send = torch.compile(send_wrapper, fullgraph=True, dynamic=True)
        
        try:
            # Execute the compiled function
            compiled_send(arg_0, arg_1, arg_2, sentinel)
            print(" Rank 0: Compiled send_object_list success")
        except Exception as e:
            print(f" Rank 0: Error during compiled send_object_list: {e}")

    elif rank == 1:
        # Receiver logic
        res_list = [None]
        try:
            dist.recv_object_list(res_list, src=0)
            print(" Rank 1: Receive success")
            # Basic assertion to check data integrity if needed
            # assert res_list[0] is not None
        except Exception as e:
            print(f" Rank 1: Error during receive: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Check for CUDA availability to determine backend
    if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
        print(f"Warning: Not enough CUDA devices available ({torch.cuda.device_count()}). Falling back to CPU if possible or failing.")
        # Forcing CPU if not enough GPUs, though nccl might fail without GPUs.
        # In a real scenario, ensure resources match.
        pass

    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)