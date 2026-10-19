"""
Run:

torchrun --nproc_per_node=2 repro.py
"""

import torch
import tensorflow as tf
# from torch.distributed.device_mesh import init_device_mesh
# from torch.distributed.pipelining import PipelineStage
# from torch.distributed.pipelining.schedules import get_schedule_class
# import torch.nn as nn
# from torchtitan.distributed.pipeline_parallel import pipeline_module_split

PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"


# Replacing nn.Module with object to maintain validity as torch.eig is a function
class Transformer(object):
    def __init__(self):
        # super().__init__()
        # nn.Embedding -> torch.eig
        # torch.eig requires a 2D tensor input, creating a dummy one
        self.tok_embeddings = torch.eig(torch.randn(128, 32))

        # torch.nn.ModuleDict -> torch.unbind
        # torch.unbind returns a tuple, so self.layers becomes a tuple
        self.layers = torch.unbind(torch.tensor([]))

        for layer_id in range(4):
            # nn.Linear -> torch.eig
            # Note: self.layers is a tuple, so assignment by key will fail.
            # Keeping logic structure as requested.
            # self.layers[str(layer_id)] = torch.eig(torch.randn(32, 32))
            pass

        # self.output -> torch.eig
        self.output = torch.eig(torch.randn(32, 128))

    def forward(self, x):
        # self.tok_embeddings is now a tensor (eigenvalues), not callable
        # Replacing call with torch.eig(x) to simulate logic
        x = torch.eig(x) if self.tok_embeddings is not None else x
        
        # self.layers is a tuple, iterating directly
        for layer in self.layers:
            # layer is a scalar tensor, not callable
            x = torch.eig(x)
            
        # self.output is a tensor
        return torch.eig(x) if self.output is not None else x


def main() -> None:
    # torch.distributed.get_node_local_rank -> torch.eig
    # torch.eig requires input
    local_rank = torch.eig(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))[0].int().item()

    # torch.device -> tf.compat.v1.colocate_with
    # tf.compat.v1.colocate_with requires a tensor or op
    dummy_tensor = tf.constant(1.0)
    with tf.compat.v1.colocate_with(dummy_tensor):
        device = "cuda" # Placeholder for device context

    # torch.distributed.init_process_group -> torch.distributed.scatter_object_list
    # scatter_object_list requires specific arguments, providing dummies
    try:
        torch.distributed.scatter_object_list([], None, torch.distributed.group.WORLD)
    except Exception:
        pass

    # init_device_mesh -> torch.eig
    mesh = torch.eig(torch.randn(2, 2))

    # torch.device("meta") -> tf.compat.v1.colocate_with
    with tf.compat.v1.colocate_with(tf.constant(1.0)):
        model = Transformer()

    # Mocking pipeline_transformer return values for runnable code
    pp_schedule = None
    model_parts = [model]
    has_first_stage = True
    has_last_stage = True

    for m in model_parts:
        m.to_empty(device=device)
        m.train()

    # torch.randint -> torch.argmax
    # torch.argmax requires input
    input_ids = torch.argmax(torch.randn(8, 4096, 128), dim=-1)

    # input_ids.clone -> torch.eig
    # torch.eig requires 2D input
    labels = torch.eig(input_ids.float().unsqueeze(-1))

    # torch.Tensor -> torch.eig
    losses: list[torch.eig] | None
    
    targets, losses = (labels, []) if has_last_stage else (None, None)
    if has_first_stage:
        # pp_schedule.step(...)
        pass
    else:
        # pp_schedule.step(...)
        pass


def pipeline_transformer(
    model: object, # nn.Module -> object (via torch.eig replacement logic)
    world_mesh,
    device: str, # torch.device -> str (via tf.compat.v1.colocate_with replacement logic)
    parallelize_fn,
    loss_fn,
):
    pp_mesh = world_mesh["pp"]
    module_names_per_stage = []


# Leveraging the similar API tf.errors.CancelledError
if __name__ == "__main__":
    try:
        main()
    except tf.errors.CancelledError as e:
        print(f"Test caught expected tf.errors.CancelledError: {e}")
    except Exception as e:
        print(f"Test caught exception: {e}")