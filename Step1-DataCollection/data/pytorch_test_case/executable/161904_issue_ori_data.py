"""
Run:

torchrun --nproc_per_node=2 repro.py
"""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import get_schedule_class
import torch.nn as nn
from torchtitan.distributed.pipeline_parallel import pipeline_module_split

PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"


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


def main() -> None:
    local_rank = torch.distributed.get_node_local_rank()
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("pp",))

    with torch.device("meta"):
        model = Transformer()

    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_transformer(
        model,
        mesh,
        device,
        parallelize_transformer,
        loss_fn,
    )
    for m in model_parts:
        m.to_empty(device=device)
        m.train()

    input_ids = torch.randint(0, 128, (8, 4096), device=device)
    labels = input_ids.clone()

    losses: list[torch.Tensor] | None
    targets, losses = (labels, []) if has_last_stage else (None, None)
    if has_first_stage:
        pp_schedule.step(
            input_ids,
            target=targets,
            losses=losses,
        )
    else:
        pp_schedule.step(
            target=targets,
            losses=losses,
        )


def pipeline_transformer(
    model: nn.Module,
    world_mesh,
    device: torch.device,
    parallelize_fn,
    loss_fn,
):
    pp_mesh = world_mesh["pp"]
    module_names_per_stage = [
        ["tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3"],
        ["norm", "output"],
    ]

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        SCHEDULE_NAME,
        device,
        module_names_per_stage,
    )

    for i, m in enumerate(model_parts):
        m = parallelize_fn(m)
        model_parts[i] = m
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(stages, loss_fn)

    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def build_pipeline_schedule(stages: list[PipelineStage], loss_fn):
    microbatch_size = 1  # job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = 8
    assert batch_size % microbatch_size == 0
    n_microbatches = batch_size // microbatch_size
    num_total_stages = PP_DEGREE * len(stages)
    assert n_microbatches >= num_total_stages
    schedule_class = get_schedule_class(SCHEDULE_NAME)
    return schedule_class(
        stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


def parallelize_transformer(model: nn.Module, compile=True) -> nn.Module:
    if compile:
        torch._functorch.config.donated_buffer = False  # why needed?
        # apply_ac(model, job_config.activation_checkpoint)
        # apply_compile(model)
        for layer_id, transformer_block in model.layers.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            model.layers.register_module(layer_id, transformer_block)
    return model


def loss_fn(inputs, targets):
    return torch.nn.functional.cross_entropy(
        inputs.view(-1, inputs.size(-1)),
        targets.view(-1),
    )


if __name__ == "__main__":
    main()