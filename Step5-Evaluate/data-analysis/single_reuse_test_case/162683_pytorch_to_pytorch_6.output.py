import torch
import torch.distributed as dist
import time
import os

# Adapted shapes for distributed testing.
# Original shapes had dim 0 = 1, which is invalid for all_to_all with world_size > 1.
# We adjust dim 0 to 2 to allow for a standard 2-process test.
# Fixed: Removed extra tuple wrapping to resolve TypeError in modulo operation.
shapes = [
     (2, 12, 10, 64),
     (2, 12, 10, 10),
]

def benchmark_all_to_all_single(input_shape, dtype=torch.float32, device="cpu", repeat=500):
    # Create input tensor
    input_tensor = torch.empty(input_shape, dtype=dtype, device=device).uniform_(0,1) * 2 - 1
    
    # Warm up
    for _ in range(100):
        _ = dist.all_to_all_single(input_tensor, group=dist.group.WORLD)
    
    # Synchronize all processes before starting the benchmark
    dist.barrier()

    # Run
    times = []
    for i in range(repeat):
        dist.barrier()
        start = time.time()
        _ = dist.all_to_all_single(input_tensor, group=dist.group.WORLD)
        dist.barrier()
        end = time.time()
        
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
    
    times.sort()
    
    # Only print from rank 0 to avoid messy output
    if dist.get_rank() == 0:
        print(times)
        
    avg_time_ms = sum(times) / len(times) if times else 0
    return avg_time_ms

def main():
    # Initialize the process group
    # Using 'gloo' backend as it is commonly available for CPU testing
    backend = 'gloo' 
    dist.init_process_group(backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running benchmark on {world_size} ranks.")

    for shape in shapes:
        # Ensure the first dimension is divisible by world_size for all_to_all
        if shape[0] % world_size != 0:
            if rank == 0:
                print(f"Skipping shape {shape}: first dimension not divisible by world_size {world_size}")
            continue

        t = benchmark_all_to_all_single(shape)
        if rank == 0:
            print(f"{shape} -> {t:.3f} us")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Setup environment variables if not set (e.g., running directly without torchrun)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
    main()