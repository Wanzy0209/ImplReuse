import torch
import torch.distributed as dist
import os

def test_recv_object_list_xpu():
    """
    Test case for torch.distributed.recv_object_list on XPU.
    Adapted from the context of Issue 159974 (XPU Segfault).
    """
    
    # Initialize the process group
    # 'ccl' is the standard backend for Intel XPU distributed training
    backend = 'ccl' if torch.xpu.is_available() else 'gloo'
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        print("This test requires at least 2 processes (world_size >= 2).")
        dist.destroy_process_group()
        return

    # Target device: XPU (as per the original bug report)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    if rank == 0:
        # Sender Process
        # Create a tensor on XPU similar to the original bug report
        x = torch.randn(128).to(device)
        obj_list = [x]
        
        print(f"Rank {rank}: Sending object list on {device}...")
        dist.send_object_list(obj_list, dst=1)
        print(f"Rank {rank}: Send passed")

    elif rank == 1:
        # Receiver Process
        obj_list = [None]
        
        print(f"Rank {rank}: Receiving object list on {device}...")
        # API Under Test: torch.distributed.recv_object_list
        # We pass the device argument to ensure placement on XPU
        dist.recv_object_list(obj_list, src=0, device=device)
        print(f"Rank {rank}: Recv passed")

        # Assertions to verify correctness
        assert obj_list[0] is not None, "Received object is None"
        assert isinstance(obj_list[0], torch.Tensor), "Received object is not a Tensor"
        
        if torch.xpu.is_available():
            assert obj_list[0].device.type == 'xpu', "Tensor not on XPU device"
        
        print(f"Rank {rank}: Assertions passed")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for distributed environment variables
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("This test is designed to run in a distributed environment.")
        print("Please run using torchrun, for example:")
        print("torchrun --nproc_per_node=2 <script_name>.py")
    else:
        test_recv_object_list_xpu()