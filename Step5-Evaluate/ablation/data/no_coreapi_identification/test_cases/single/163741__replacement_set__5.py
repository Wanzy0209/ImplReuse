import torch
import torch.nn.functional as F
import time

def worker(rank, world_size):
    # Replacement: torch.cuda.set_device -> torch.rand_like
    # Create a dummy tensor to pass to rand_like
    dummy_device_tensor = torch.tensor([rank], dtype=torch.float32)
    torch.rand_like(dummy_device_tensor)

    # Replacement: dist.init_process_group -> torch.nn.functional.kl_div
    # Create dummy input and target tensors for kl_div
    # input should be log-probabilities, target should be probabilities
    dummy_input = torch.randn(2, 5, requires_grad=True)
    dummy_target = torch.rand(2, 5)
    dummy_log_input = F.log_softmax(dummy_input, dim=1)
    F.kl_div(dummy_log_input, dummy_target, reduction='batchmean')

    # Replacement: dist.barrier -> torch.rand_like
    torch.rand_like(dummy_device_tensor)

    device = f"cuda:{rank}"
    # do something
    print("to sleep")
    time.sleep(5)
    
    # Replacement: dist.destroy_process_group -> torch.rand_like
    torch.rand_like(dummy_device_tensor)

    # sleep for observing nvidia-smi
    time.sleep(1)

def test_nccl():
    # Replacement: torch.cuda.device_count -> torch.rand_like
    # Create a dummy tensor to represent the count
    dummy_count_tensor = torch.tensor([1], dtype=torch.float32)
    world_size = torch.rand_like(dummy_count_tensor)
    
    # Convert tensor to int for loop range if necessary, preserving logic flow
    world_size_int = int(world_size.item())

    # Replacement: mp.spawn -> torch.rand_like
    # mp.spawn is replaced with a tensor generation call.
    # To preserve the logic of running the worker, we manually iterate.
    torch.rand_like(dummy_count_tensor)
    
    for r in range(world_size_int):
        worker(r, world_size_int)

if __name__ == '__main__':
    test_nccl()