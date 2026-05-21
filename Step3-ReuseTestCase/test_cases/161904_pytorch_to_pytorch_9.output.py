"""
Run:
torchrun --nproc_per_node=2 test_case.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import os

def main():
    # Setup similar to the bug report
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    # --- Test for torch.distributed.broadcast_object_list ---
    
    # Prepare data: Rank 0 has the schedule info, others have placeholders
    if dist.get_rank() == 0:
        schedule_info = [{"name": "ZBVZeroBubble", "degree": 2}]
    else:
        schedule_info = [None]

    # Call the API
    dist.broadcast_object_list(schedule_info, src=0)

    # Verify the broadcast
    assert schedule_info[0]["name"] == "ZBVZeroBubble"
    assert schedule_info[0]["degree"] == 2
    
    print(f"Rank {dist.get_rank()} successfully broadcasted object: {schedule_info[0]}")

    # --- Context from bug report: torch.compile ---
    # Ensure the environment is stable for compiled models after broadcast
    class SimpleModel(nn.Module):
        def forward(self, x):
            return x + 1

    model = SimpleModel().to(device)
    compiled_model = torch.compile(model)
    
    input_tensor = torch.randn(2, 2).to(device)
    output = compiled_model(input_tensor)
    
    assert output.shape == input_tensor.shape
    print(f"Rank {dist.get_rank()} compiled model run successful.")

if __name__ == "__main__":
    main()