import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm_mem

def run_rendezvous():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.init()  # Ensure CUDA context
    
    buffer = torch_symm_mem.empty(64, device=device, dtype=torch.bfloat16)
    handle = torch_symm_mem.rendezvous(buffer, dist.group.WORLD.group_name)