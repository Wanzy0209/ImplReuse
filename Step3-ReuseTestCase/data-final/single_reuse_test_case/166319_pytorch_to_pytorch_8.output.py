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
        
        # Send the objects to rank 1
        dist.send_object_list(objects, dst=1)
        print("Rank 0: Sent objects successfully.")

    elif rank == 1:
        # --- Receiver Process ---
        # Define the function that includes the similar API call (recv_object_list)
        # and the logic from the original fuzzed program.
        def recv_and_process():
            # Prepare list to receive objects
            obj_list = [None, None, None, None]
            
            # Call the similar API: torch.distributed.recv_object_list
            dist.recv_object_list(obj_list, src=0)
            
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

        # Attempt to compile the function containing recv_object_list
        # This mimics the original bug scenario where torch.compile was used
        try:
            compiled_program = torch.compile(recv_and_process, fullgraph=True, dynamic=True)
            result = compiled_program()
            print(f"Rank 1: Received and processed result. Shape: {result.shape}")
        except Exception as e:
            print(f"Rank 1: Error during compiled execution: {e}")

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