import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import os

def worker(rank, world_size):
    # Setup from the bug report
    # Fix: Use the rank passed by mp.spawn instead of os.environ
    gpu_id = rank
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Initialize process group
    # Note: We set MASTER_ADDR and MASTER_PORT in the main function
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    # Adaptation: Test torch.nn.functional.cross_entropy
    # Create dummy tensors on the current device
    batch_size = 4
    num_classes = 10
    input_tensor = torch.randn(batch_size, num_classes).cuda(gpu_id)
    target_tensor = torch.randint(0, num_classes, (batch_size,)).cuda(gpu_id)

    # Call the similar API
    loss = F.cross_entropy(input_tensor, target_tensor)

    # Assertion to verify execution
    assert not torch.isnan(loss), "Loss is NaN"
    
    if dist.get_rank() == 0:
        print(f"Cross entropy test passed. Loss: {loss.item()}")

    dist.destroy_process_group()

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping test.")
        return

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs available. Skipping test.")
        return

    # Set environment variables for the 'env://' init method
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()