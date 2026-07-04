import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo", # Use gloo for compatibility
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{29500}"
    )

    # Determine device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        # --- Sender Process ---
        # Replicate the tensor generation from the original bug report
        torch.manual_seed(1215252001)
        
        arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1)).to(device)
        arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1)).to(device)
        arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1)).to(device)
        sentinel = torch.tensor(1.0, requires_grad=True).to(device)

        # Pack arguments into a list to send
        objects = [arg_0, arg_1, arg_2, sentinel]
        
        # Fix: send_object_list is not available in older PyTorch versions.
        # Use dist.send in a loop instead.
        for obj in objects:
            dist.send(obj, dst=1)
        
        print("Rank 0: Sent objects successfully.")

    elif rank == 1:
        # --- Receiver Process ---
        # Define the function that includes the similar API call (recv_object_list)
        # and the logic from the original fuzzed program.
        def recv_and_process():
            # Fix: recv_object_list is not available. 
            # We must allocate tensors with correct shapes and dtypes to receive data.
            # Shapes and dtypes are inferred from the Sender logic.
            
            # arg_0: int32, (9, 1, 15, 4)
            recv_0 = torch.empty((9, 1, 15, 4), dtype=torch.int32, device=device)
            dist.recv(recv_0, src=0)
            
            # arg_1: int64, (20, 15)
            recv_1 = torch.empty((20, 15), dtype=torch.int64, device=device)
            dist.recv(recv_1, src=0)
            
            # arg_2: int64, (18, 15)
            recv_2 = torch.empty((18, 15), dtype=torch.int64, device=device)
            dist.recv(recv_2, src=0)
            
            # sentinel: float, scalar, requires_grad=True
            recv_3 = torch.empty((), dtype=torch.float, device=device)
            dist.recv(recv_3, src=0)
            recv_3.requires_grad_(True)
            
            obj_list = [recv_0, recv_1, recv_2, recv_3]
            
            arg_0, arg_1, arg_2, sentinel = obj_list

            # Original logic from the fuzzed_program
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
            
            # Ensure gradient computation by multiplying with sentinel and taking real part
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        # Set seed for consistency in the receiver logic (torch.randint)
        torch.manual_seed(1215252001)

        # Fix: torch.compile is not available in older PyTorch versions.
        # Check availability before attempting to compile.
        if hasattr(torch, 'compile'):
            try:
                compiled_program = torch.compile(recv_and_process, fullgraph=True, dynamic=True)
                result = compiled_program()
                print(f"Rank 1: Received and processed result (compiled). Shape: {result.shape}")
            except Exception as e:
                print(f"Rank 1: Error during compiled execution: {e}")
        else:
            print("Rank 1: torch.compile not available, running eager mode.")
            try:
                result = recv_and_process()
                print(f"Rank 1: Received and processed result (eager). Shape: {result.shape}")
            except Exception as e:
                print(f"Rank 1: Error during eager execution: {e}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Check if CUDA is available to inform the user
    if torch.cuda.is_available():
        print("CUDA detected. Running on CUDA.")
    else:
        print("CUDA not detected. Running on CPU.")

    # Spawn processes for distributed testing
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)