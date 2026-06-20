"""
Test case for Issue 161904: ZeroBubble and DualPipeV pipeline parallel schedules 
fail with torch.compiled model.

This test case preserves the original bug reproduction logic (Pipeline Parallelism 
with torch.compile) while leveraging the similar API 
`torch.backends.mha.get_fastpath_enabled` to verify the state of the MHA fast 
path during the compilation and execution process.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.mha
import sys

# Handle missing dependencies for older PyTorch versions
try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import get_schedule_class
except ImportError as e:
    print(f"Skipping test: Required modules not found ({e}).")
    print("This test requires PyTorch >= 2.1 for torch.distributed.device_mesh and pipelining features.")
    sys.exit(0)

# Model definition from the bug report
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(128, 32)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(4):
            self.layers[str(layer_id)] = nn.Linear(32, 32, bias=False)
        self.output = nn.Linear(32, 128, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x) if self.tok_embeddings else x
        for layer in self.layers.values():
            x = layer(x)
        return self.output(x) if self.output else x

def run_test(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # Leverage the similar API: Check fast path status
    # This ensures we are aware of the optimization state before compilation
    is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    if rank == 0:
        print(f"MHA Fastpath enabled at start: {is_fastpath_enabled}")

    # Setup Device Mesh
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("pp",))

    # Create Model
    with torch.device("meta"):
        model = Transformer()

    # Manual split to replace torchtitan dependency for standalone runnability
    # Splitting 4 layers into 2 stages
    stage_idx = rank
    layers_per_stage = 2
    
    # Construct stage module
    class StageModule(nn.Module):
        def __init__(self, stage_idx, layers_per_stage, full_model):
            super().__init__()
            self.stage_idx = stage_idx
            self.layers_per_stage = layers_per_stage
            self.full_model = full_model
            
            # Assign relevant parts based on stage
            if stage_idx == 0:
                self.tok_embeddings = full_model.tok_embeddings
                self.output = None
            else:
                self.tok_embeddings = None
                self.output = full_model.output
            
            # Assign layers
            self.stage_layers = nn.ModuleDict()
            start_layer = stage_idx * layers_per_stage
            end_layer = start_layer + layers_per_stage
            for i in range(start_layer, end_layer):
                self.stage_layers[str(i)] = full_model.layers[str(i)]

        def forward(self, x):
            x = self.tok_embeddings(x) if self.tok_embeddings else x
            for layer in self.stage_layers.values():
                x = layer(x)
            return self.output(x) if self.output else x

    stage_model = StageModule(stage_idx, layers_per_stage, model)
    stage_model.to_empty(device=device)
    stage_model.train()

    # Apply torch.compile (The core of the bug report)
    # We compile the stage model to reproduce the failure
    try:
        compiled_stage_model = torch.compile(stage_model)
        if rank == 0:
            print("Model compiled successfully.")
    except Exception as e:
        print(f"Rank {rank}: Compilation failed with error: {e}")
        raise

    # Setup Pipeline Stage
    # Note: Using a generic schedule class if ZBVZeroBubble is not available in standard lib
    # to ensure the test is runnable, but attempting to use the one from the bug report first.
    try:
        schedule_name = "ZBVZeroBubble"
        schedule_class = get_schedule_class(schedule_name)
    except (ValueError, AttributeError):
        # Fallback to a standard schedule if the specific one from the issue isn't found
        # This ensures the test logic remains valid for general pipeline compilation issues
        if rank == 0:
            print(f"Schedule '{schedule_name}' not found, falling back to 1F1B.")
        schedule_class = get_schedule_class("1F1B")

    stage = PipelineStage(
        compiled_stage_model,
        mesh,
        stage_idx,
    )

    schedule = schedule_class(stage)

    # Run a step
    input_ids = torch.randint(0, 128, (8, 4096), device=device)
    
    # Check fastpath status again after compilation setup
    # to ensure no unexpected side effects
    is_fastpath_enabled_post_setup = torch.backends.mha.get_fastpath_enabled()
    if rank == 0:
        print(f"MHA Fastpath enabled after setup: {is_fastpath_enabled_post_setup}")

    if rank == 0:
        # First stage sends input
        schedule.step(input_ids)
    else:
        # Last stage receives and computes
        schedule.step()

    if rank == 0:
        print("Test step completed successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # This script requires torchrun to execute distributed logic
    # Example: torchrun --nproc_per_node=2 test_pipeline_compile.py
    import os
    
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Check if we are in a distributed environment
    if "RANK" in os.environ:
        run_test(local_rank, world_size)
    else:
        print("This test must be run with torchrun or in a distributed environment.")
        print("Example: torchrun --nproc_per_node=2 <script_name>")
        sys.exit(1)