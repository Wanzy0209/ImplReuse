"""
Test case for Issue 161904 adapted to use torch.linalg.solve.
This test verifies that torch.compile works with pipeline parallel schedules
when the model uses linear algebra operations (specifically torch.linalg.solve,
the successor to the deprecated torch.solve).

Run:
torchrun --nproc_per_node=2 test_solve_pipeline.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist

# Handle missing dependencies for older PyTorch versions
try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import get_schedule_class
except ImportError:
    print("Skipping test: torch.distributed.device_mesh requires PyTorch >= 2.1.")
    sys.exit(0)

PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"


class SolveLayer(nn.Module):
    """
    A custom layer that uses torch.linalg.solve to perform a linear operation.
    This leverages the similar API (torch.solve/torch.linalg.solve) pattern.
    """
    def __init__(self, dim=32):
        super().__init__()
        # Create a learnable matrix A for the operation Ax = b
        # Initialize with Identity + noise to ensure it is invertible
        self.A = nn.Parameter(torch.eye(dim) + 0.1 * torch.randn(dim, dim))

    def forward(self, x):
        # x shape: (Batch, Dim)
        # A shape: (Dim, Dim)
        
        # To use torch.linalg.solve with batches, we expand A to match the batch size
        batch_size = x.size(0)
        A_batch = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape x to (Batch, Dim, 1) to act as the B matrix in AX = B
        x_unsqueezed = x.unsqueeze(-1)
        
        # Solve for X: A * X = x
        # This replaces the standard Linear layer logic with a solve operation
        y = torch.linalg.solve(A_batch, x_unsqueezed)
        
        # Return shape (Batch, Dim)
        return y.squeeze(-1)


class TransformerWithSolve(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(128, 32)
        # Replace standard Linear layers with SolveLayer to test the similar API
        self.layers = nn.ModuleList([SolveLayer(32) for _ in range(4)])
        self.output = nn.Linear(32, 128, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    mesh = init_device_mesh("cuda", (PP_DEGREE,), mesh_dim_names=("pp",))

    # Initialize model on meta device to save memory during setup
    with torch.device("meta"):
        model = TransformerWithSolve()

    # Manually split the model into stages to avoid external dependencies like torchtitan
    # Stage 0: Embeddings + First 2 SolveLayers
    class Stage0(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.embed = base_model.tok_embeddings
            self.layers = base_model.layers[:2]

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return x

    # Stage 1: Last 2 SolveLayers + Output
    class Stage1(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.layers = base_model.layers[2:]
            self.output = base_model.output

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.output(x)

    # Instantiate stages
    with torch.device("meta"):
        stage0_mod = Stage0(model)
        stage1_mod = Stage1(model)

    # Move stages to actual device
    stage0_mod.to_empty(device=device)
    stage1_mod.to_empty(device=device)
    stage0_mod.train()
    stage1_mod.train()

    # Apply torch.compile to the stages
    # This is the core of the bug reproduction: compiling models with specific schedules
    print(f"Rank {local_rank}: Compiling stage...")
    stage0_mod = torch.compile(stage0_mod)
    stage1_mod = torch.compile(stage1_mod)

    # Assign stage based on rank
    my_stage_idx = mesh.get_local_rank("pp")
    stage_mod = stage0_mod if my_stage_idx == 0 else stage1_mod

    # Create PipelineStage
    stage = PipelineStage(
        stage_mod,
        mesh,
        stage_idx=my_stage_idx
    )

    # Get Schedule
    schedule_cls = get_schedule_class(SCHEDULE_NAME)
    schedule = schedule_cls(stage, mesh)

    # Run a step
    input_ids = torch.randint(0, 128, (8, 4096), device=device)
    labels = input_ids.clone()

    losses = [] if my_stage_idx == 1 else None
    targets = labels if my_stage_idx == 1 else None

    print(f"Rank {local_rank}: Running step...")
    if my_stage_idx == 0:
        schedule.step(input_ids, target=targets, losses=losses)
    else:
        schedule.step(target=targets, losses=losses)
    
    print(f"Rank {local_rank}: Step completed successfully.")


if __name__ == "__main__":
    main()