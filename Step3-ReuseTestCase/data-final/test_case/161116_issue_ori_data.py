import torch
import torch.distributed as dist
import os
def main():
    gpu_id = int(os.environ["LOCAL_RANK"])
    
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl',
                            device_id=gpu_id,
                            )
    dist.barrier()
    
if __name__ == "__main__":
    main()