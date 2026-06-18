import torch
import torch.nn as nn
import torch.distributed as dist
import os
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import get_schedule_class

# Translating semantics of tf.errors.CancelledError for cross-library reuse
class CancelledError(RuntimeError):
    """Raised when the pipeline operation is cancelled or fails unexpectedly."""
    pass

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Initialize distributed environment
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Initialize device mesh
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("pp",))
    
    # Define a simple Transformer-like model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = SimpleModel().to(device)
    
    # Split model for pipeline parallelism
    # Rank 0: layers 0-1, Rank 1: layers 2-3
    stage_idx = rank
    num_stages = world_size
    layers_per_stage = len(model.layers) // num_stages
    
    sub_layers = model.layers[stage_idx * layers_per_stage : (stage_idx + 1) * layers_per_stage]
    stage_module = nn.Sequential(*sub_layers)
    
    # Apply torch.compile to the stage module
    # This is the critical part of the bug report
    try:
        compiled_module = torch.compile(stage_module)
    except Exception as e:
        raise CancelledError(f"Compilation failed on rank {rank}: {e}")

    # Create PipelineStage
    stage = PipelineStage(
        compiled_module,
        mesh,
        stage_id=rank,
        num_stages=num_stages,
        device=device
    )
    
    # Get Schedule
    # Testing "ZBVZeroBubble" as mentioned in the bug report
    schedule_name = "ZBVZeroBubble" 
    try:
        schedule_cls = get_schedule_class(schedule_name)
        schedule = schedule_cls(stage, mesh)
    except Exception as e:
        # If schedule class is not found, we might be in an env without it, 
        # but for the purpose of the bug report, we assume it exists.
        raise RuntimeError(f"Could not retrieve schedule {schedule_name}: {e}")

    # Run a step
    try:
        if rank == 0:
            inputs = torch.randn(8, 32, device=device)
            schedule.step(inputs)
        else:
            schedule.step()
    except Exception as e:
        raise CancelledError(f"Pipeline step failed on rank {rank}: {e}")

    print(f"Rank {rank} completed successfully with {schedule_name} and torch.compile.")

if __name__ == "__main__":
    main()