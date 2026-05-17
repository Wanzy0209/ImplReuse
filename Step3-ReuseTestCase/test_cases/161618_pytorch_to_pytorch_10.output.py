import torch
import torch.distributed as dist
import os

def main():
    """
    Test case for torch.distributed.gather_object.
    Adapted from the structure of the original torch.compile bug report.
    """
    # Initialize the distributed process group
    # Using 'gloo' for CPU compatibility or 'nccl' if CUDA is available
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Setup: Create a tensor object similar to the original script
    # Original: a = torch.randn((m, n)).requires_grad_(False).cuda()
    # Adaptation: Create a unique tensor per rank to verify gathering
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Create a simple object (tensor) to gather
    obj_to_gather = torch.tensor([rank], device=device)

    # Prepare the output list (only required on the destination rank)
    if rank == 0:
        gathered_objects = [None] * world_size
    else:
        gathered_objects = None

    # API Call: torch.distributed.gather_object
    # Replaces the original 'compiled(a, mat1, mat2)' call site
    dist.gather_object(
        obj=obj_to_gather,
        object_gather_list=gathered_objects,
        dst=0
    )

    # Verification: Check if the destination rank received all objects correctly
    if rank == 0:
        for i in range(world_size):
            assert gathered_objects[i].item() == i, f"Expected {i}, got {gathered_objects[i].item()}"
        print("Test passed: torch.distributed.gather_object successful.")

if __name__ == "__main__":
    # Check if the script is being run in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        main()
    else:
        print("This test case requires a distributed environment.")
        print("Please run using torchrun, for example:")
        print("torchrun --nproc_per_node=2 <script_name>.py")