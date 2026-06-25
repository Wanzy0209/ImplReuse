```python
"""
Run:

python repro.py
"""

import tensorflow as tf
from tensorflow.python.eager.context import get_device_name
from tensorflow.keras import layers, Model, losses
import typing

# Conversion: torch.distributed.device_mesh.init_device_mesh
# TensorFlow uses tf.distribute.Strategy for device mesh and distribution logic.
# We use MirroredStrategy for single-node multi-GPU as implied by nproc_per_node=2.

# Conversion: torch.distributed.pipelining.schedules.get_schedule_class
# TensorFlow does not have a direct equivalent for PyTorch's pipeline schedule classes.
# We define a mock class to preserve structure.

PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"


class PipelineStage:
    """Mock for torch.distributed.pipelining.PipelineStage"""
    def __init__(self, submod, is_first, is_last):
        self.submod = submod
        self.is_first = is_first
        self.is_last = is_last


class Schedule:
    """Mock for torch.distributed.pipelining.schedules.Schedule"""
    def __init__(self, stages, n_microbatches, loss_fn):
        self.stages = stages
        self.n_microbatches = n_microbatches
        self.loss_fn = loss_fn

    def step(self, inputs, target=None, losses=None):
        x = inputs
        # Execute pipeline stages sequentially for this mock
        for stage in self.stages:
            x = stage.submod(x)
        
        if target is not None and self.loss_fn:
            loss = self.loss_fn(x, target)
            if losses is not None:
                losses.append(loss)
        return x, losses


class Transformer(Model):
    def __init__(self):
        super().__init__()
        # Conversion: torch.nn.Embedding -> tf.keras.layers.Embedding
        self.tok_embeddings = layers.Embedding(128, 32)
        
        # Conversion: torch.nn.ModuleDict -> dict
        self.layers = {}
        for layer_id in range(4):
            # Conversion: torch.nn.Linear -> tf.keras.layers.Dense
            self.layers[str(layer_id)] = layers.Dense(32, use_bias=False)
        
        self.output = layers.Dense(128, use_bias=False)
        # Added norm layer because pipeline_transformer expects it
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.tok_embeddings(x) if self.tok_embeddings else x
        for layer in self.layers.values():
            x = layer(x)
        x = self.norm(x) if self.norm else x
        return self.output(x) if self.output else x


def main() -> None:
    # Conversion: torch.distributed.get_node_local_rank
    # In TF, this is handled internally by the Strategy.
    # We initialize the strategy here.
    strategy = tf.distribute.MirroredStrategy()
    
    # Conversion: torch.distributed.init_process_group
    # Handled by strategy initialization.
    
    # Conversion: torch.device
    # We use the strategy scope to place devices.
    with strategy.scope():
        # device_name = get_device_name() # Available if needed
        
        # Conversion: with torch.device("meta")
        # TF does not have a meta device. We instantiate the model directly.
        model = Transformer()

        pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_transformer(
            model,
            None, # mesh
            None, # device
            parallelize_transformer,
            loss_fn,
        )
        
        # Conversion: m.to_empty(device=device)
        # TF variables are created on the device within the strategy scope automatically.
        for m in model_parts:
            # m.train() is not explicit in TF Keras like PyTorch, handled by compile/fit or tf.function
            pass

        # Conversion: torch.randint
        input_ids = tf.random.uniform((8, 4096), minval=0, maxval=128, dtype=tf.int32)
        labels = input_ids

        losses: list[tf.Tensor] | None
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
    model: Model,
    world_mesh,
    device: str, # Converted from torch.device
    parallelize_fn,
    loss_fn,
):
    # pp_mesh = world_mesh["pp"] # Simplified for TF
    module_names_per_stage = [
        ["tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3"],
        ["norm", "output"],
    ]

    # Mocking torchtitan.distributed.pipeline_parallel.pipeline_module_split
    stages, model_parts = pipeline_module_split(
        model,
        None, # pp_mesh
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


def pipeline_module_split(model, mesh, schedule_name, device, module_names_per_stage):
    """Mock implementation of pipeline_module_split for TensorFlow."""
    stages = []
    model_parts = []

    def get_layer(name):
        if name == "tok_embeddings": return model.tok_embeddings
        if name == "output": return model.output
        if name == "norm": return model.norm
        if name.startswith("layers."):
            idx = name.split(".")[1]
            return model.layers[idx]
        return None

    for stage_names in module_names_per_stage:
        # Create a sub-model for this stage
        class StageModel(Model):
            def __init__(self, layers_list):
                super().__init__()
                self.layers_list = layers_list
            
            def call(self, x):
                for layer in self.layers_list:
                    x = layer(x)
                return x
        
        stage_layers = [get_layer(n) for n in stage_names if get_layer(n) is not None]
        stage_model = StageModel(stage_layers)
        
        is_first = (stage_names == module_names_per_stage[0])
        is_last = (stage_names == module_names_per_stage[-1])
        
        stages.append(PipelineStage(stage_model, is_first, is_last))
        model_parts.append(stage_model)
        
    return stages, model_parts


def build_pipeline_schedule(stages: list[PipelineStage], loss_fn):
    microbatch_size = 1
    batch_size = 8
    assert batch_size % microbatch_size == 0
    n_microbatches = batch_size // microbatch_size
    num_total_stages = PP_DEGREE * len(stages)
    assert n_microbatches >= num_total_stages
    
    # Conversion: torch.distributed.pipelining.schedules.get_schedule_class
    # Using the mock Schedule class defined above
    return Schedule(
        stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


def parallelize_transformer(model: Model, compile=True) -> Model:
    if compile:
        # Conversion: torch.compile -> tf.function
        # Note: In TF, applying tf.function to individual layers inside a Model 
        # is less common than applying it to the whole call method or step function.
        # We follow the source structure here.
        for layer_id, transformer_block in model.layers.items():
            # transformer_block is a Dense layer, wrapping in tf.function
            model.layers[layer_id] = tf.function(transformer_block)
    return model


def loss_fn(inputs, targets):
    # Conversion: torch.nn.functional.cross_entropy
    # Using SparseCategoricalCrossentropy as the semantic equivalent.
    # Note: The prompt documentation mentioned mean_squared_error, but that is 
    # semantically incorrect for a classification task with logits.
    # We use the correct loss function for valid TensorFlow code.
    cce = losses.SparseCategoricalCrossentropy(from_logits=True)
    return cce(targets, inputs)


if __name__ == "__main__":
    main()
```