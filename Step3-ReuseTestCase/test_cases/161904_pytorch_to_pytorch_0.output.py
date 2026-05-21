"""
Test case for Issue 161904: ZeroBubble and DualPipeV pipeline parallel schedules fail with torch.compiled model.

This script verifies the interaction between torch.compile and specific pipeline parallel schedules.
It replaces the dependency on torchtitan with a manual pipeline split to ensure the test is runnable
with standard PyTorch.

Run:
torchrun --nproc_per_node=2 test_torch_compile_pipeline.py
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import get_schedule_class

# Configuration
PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # Can be switched to "DualPipeV" to test that schedule


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(128, 32)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(4):
            self.layers.append(nn.Linear(32, 32, bias=False))
        self.output = nn.Linear(32, 128, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x) if self.tok_embeddings else x
        for layer in self.layers:
            x = layer(x)
        return self.output(x) if self.output else x


def main() -> None:
    # Initialize Distributed Environment
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Create Device Mesh for Pipeline Parallelism
    mesh = init_device_mesh("cuda", (PP_DEGREE,), mesh_dim_names=("pp",))
    pp_mesh = mesh["pp"]
    stage_idx = pp_mesh.get_local_rank()

    # Initialize Model
    # Note: We initialize on CPU/Meta first to avoid OOM on all ranks before splitting
    with torch.device("meta"):
        model = Transformer()

    # Manual Pipeline Split (replacing torchtitan.distributed.pipeline_parallel)
    # Split the 4 layers into 2 stages
    layers = list(model.layers.children())
    mid_point = len(layers) // 2

    if stage_idx == 0:
        # Stage 0: Embedding + First half of layers
        class Stage0Module(nn.Module):
            def __init__(self, embed, layer_list):
                super().__init__()
                self.embed = embed
                self.layers = nn.ModuleList(layer_list)

            def forward(self, x):
                x = self.embed(x)
                for layer in self.layers:
                    x = layer(x)
                return x

        submod = Stage0Module(model.tok_embeddings, layers[:mid_point])
    else:
        # Stage 1: Second half of layers + Output head
        class Stage1Module(nn.Module):
            def __init__(self, layer_list, output):
                super().__init__()
                self.layers = nn.ModuleList(layer_list)
                self.output = output

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)

        submod = Stage1Module(layers[mid_point:], model.output)

    # Move sub-module to the correct device and initialize weights
    submod.to_empty(device=device)
    # Reset parameters manually since to_empty doesn't initialize weights
    for p in submod.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)
    
    submod.train()

    # --- API Under Test: torch.compile ---
    # The bug report indicates that compiling the model parts causes issues with specific schedules.
    # We apply torch.compile here to verify the behavior.
    compiled_submod = torch.compile(submod)

    # Create PipelineStage
    stage = PipelineStage(
        compiled_submod,
        pp_mesh,
        stage_idx,
    )

    # Get Schedule
    schedule_cls = get_schedule_class(SCHEDULE_NAME)
    schedule = schedule_cls(stage, pp_mesh)

    # Prepare Data
    # Only rank 0 has input, only rank 1 has target (for loss calculation)
    input_ids = torch.randint(0, 128, (8, 4096), device=device) if stage_idx == 0 else None
    labels = torch.randint(0, 128, (8, 4096), device=device) if stage_idx == 1 else None

    # Define a simple loss function for the last stage
    def loss_fn(outputs, targets):
        return nn.functional.cross_entropy(outputs.view(-1, 128), targets.view(-1))

    # Run a step
    # Note: ZBVZeroBubble and DualPipeV are 1F1B style schedules.
    # We pass the loss function to the step if it's the last stage.
    
    try:
        if stage_idx == 0:
            schedule.step(input_ids)
        else:
            # For simplicity in this repro, we just run forward. 
            # A full training step would require passing the loss_fn and handling returns.
            # However, the bug report mentions failure with the schedule itself.
            # We call step with target to simulate the backward pass trigger if supported by the schedule impl.
            schedule.step(target=labels)
        
        # If we reach here without error, the test passes for this configuration.
        if rank == 0:
            print(f"Successfully ran {SCHEDULE_NAME} schedule with torch.compile.")
    except Exception as e:
        if rank == 0:
            print(f"Error running {SCHEDULE_NAME} with torch.compile: {e}")
        raise e


if __name__ == "__main__":
    main()