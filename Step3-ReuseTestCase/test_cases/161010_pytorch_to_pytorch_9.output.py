import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_broadcast_preserves_stride(rank, world_size):
    """
    Test case to verify if torch.distributed.broadcast_object_list 
    preserves the stride of tensors, similar to the stride preservation 
    issue observed with torch.compile and clone(memory_format=torch.preserve_format).
    """
    # Setup for distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Replicate the tensor creation logic from the original bug report
    # to generate a tensor with specific stride characteristics
    if rank == 0:
        A = torch.rand(5, 5)
        Q, R = torch.linalg.qr(A)
        rhs = torch.ones(Q.shape[0], 1, device=A.device)
        a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
        
        # Store original stride for comparison
        original_stride = a.stride()
        obj_list = [a]
    else:
        obj_list = [None]
        original_stride = None

    # The API call to test: torch.distributed.broadcast_object_list
    # This replaces the 'torch.compile(f)' call site in terms of being the operation under test
    dist.broadcast_object_list(obj_list, src=0)

    # Verification
    if rank == 0:
        received_tensor = obj_list[0]
        received_stride = received_tensor.stride()
        
        # The original bug checked: a.stride() == a.clone(...).stride()
        # Here we check if the broadcast operation preserved the stride
        if original_stride == received_stride:
            print("Test Passed: Stride preserved by broadcast_object_list")
        else:
            print(f"Test Failed: Stride mismatch. Original: {original_stride}, Received: {received_stride}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_broadcast_preserves_stride, args=(world_size,), nprocs=world_size, join=True)