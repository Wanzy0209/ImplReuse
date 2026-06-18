import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import get_schedule_class

# Import the similar API identified in the analysis
import torch.special

class SimpleModel(nn.Module):
    """
    A minimal model incorporating the similar API 
    (torch.special.scaled_modified_bessel_k0) to test compilation 
    within a pipeline parallel context.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        # Leverage the similar API: scaled_modified_bessel_k0
        # We add 1.0 and take abs to ensure input is positive, 
        # as this function is sensitive to input domain.
        x = torch.special.scaled_modified_bessel_k0(torch.abs(x) + 1.0)
        x = self.layer2(x)
        return x

def main():
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Setup device mesh for pipeline parallelism
    # Assuming 2 GPUs for this test (PP_DEGREE = 2)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("pp",))

    # Create the full model
    model = SimpleModel().to(device)

    # Manually split the model into stages to avoid external dependencies (like torchtitan)
    # Stage 0: layer1 + special_op
    # Stage 1: layer2
    if rank == 0:
        stage_module = nn.Sequential(model.layer1, 
                                     lambda x: torch.special.scaled_modified_bessel_k0(torch.abs(x) + 1.0))
    else:
        stage_module = model.layer2

    # Move stage module to device
    stage_module = stage_module.to(device)
    stage_module.train()

    # Apply torch.compile to the pipeline stage
    # This is the core condition for the reported bug (Issue 161904)
    print(f"Rank {rank}: Compiling stage...")
    compiled_stage = torch.compile(stage_module)

    # Initialize PipelineStage
    stage = PipelineStage(
        compiled_stage,
        mesh,
        rank,
    )

    # Initialize the schedule mentioned in the bug report
    schedule_name = "ZBVZeroBubble" 
    schedule = get_schedule_class(schedule_name)(
        stage,
        n_microbatches=4, # Use a small number for minimal testing
    )

    # Run a single step
    if rank == 0:
        # Create dummy input
        x = torch.randn(2, 10, device=device)
        schedule.step(x)
    else:
        schedule.step()

    print(f"Rank {rank}: Step completed successfully with {schedule_name} and torch.compile.")

if __name__ == "__main__":
    main()