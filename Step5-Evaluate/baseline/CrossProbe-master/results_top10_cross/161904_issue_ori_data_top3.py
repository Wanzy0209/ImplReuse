```python
"""
Run:

python repro.py
(Note: TensorFlow uses strategies for distribution, not torchrun.
This code uses MirroredStrategy to simulate the multi-device setup.)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import typing

# Conversion: Constants
PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"

# Conversion: Mocking torch.distributed.pipelining.PipelineStage
class PipelineStage:
    def __init__(self, submod, is_first=False, is_last=False):
        self.submod = submod
        self.is_first = is_first
        self.is_last = is_last

# Conversion: Mocking torchtitan.distributed.pipeline_parallel.pipeline_module_split
def pipeline_module_split(model, pp_mesh, schedule_name, device, module_names_per_stage):
    """
    Splits the Keras model into stages based on module names.
    In TensorFlow, we create sub-models for each stage.
    """
    stages = []
    model_parts = []
    
    # Helper to get layer by name (handling 'layers.0' syntax)
    def get_layer(model, name):
        if '.' in name:
            parts = name.split('.')
            obj = model
            for p in parts:
                if p.isdigit():
                    # Handle ModuleDict indexing
                    obj = obj.layers[int(p)]
                else:
                    obj = getattr(obj, p)
            return obj
        return getattr(model, name)

    for i, names in enumerate(module_names_per_stage):
        # Create a new Model for this stage
        # Note: In a real distributed TF setup, these would be placed on specific devices
        # using tf.device scopes.
        
        class StageModel(Model):
            def __init__(self, layers_list):
                super().__init__()
                self.stage_layers = layers_list
            
            def call(self, x):
                for layer in self.stage_layers:
                    x = layer(x)
                return x

        stage_layers = [get_layer(model, name) for name in names]
        stage_model = StageModel(stage_layers)
        
        is_first = (i == 0)
        is_last = (i == len(module_names_per_stage) - 1)
        
        stage = PipelineStage(stage_model, is_first, is_last)
        stages.append(stage)
        model_parts.append(stage_model)

    return stages, model_parts

# Conversion: Mocking torch.distributed.pipelining.schedules.get_schedule_class
def get_schedule_class(schedule_name):
    class PipelineSchedule:
        def __init__(self, stages, n_microbatches, loss_fn):
            self.stages = stages
            self.n_microbatches = n_microbatches
            self.loss_fn = loss_fn

        def step(self, inputs=None, target=None, losses=None):
            # Simplified pipeline step execution for demonstration
            # In a real implementation, this would handle microbatches and communication
            x = inputs
            for stage in self.stages:
                x = stage.submod(x)
            
            if target is not None and self.loss_fn is not None:
                loss = self.loss_fn(x, target)
                if losses is not None:
                    losses.append(loss)
            
            return x, losses

    return PipelineSchedule

# Conversion: torch.nn.Module -> tf.keras.Model
class Transformer(Model):
    def __init__(self):
        super().__init__()
        # Conversion: nn.Embedding -> tf.keras.layers.Embedding
        self.tok_embeddings = layers.Embedding(128, 32)
        
        # Conversion: nn.ModuleDict -> dict of layers
        self.layer_dict = {}
        for layer_id in range(4):
            # Conversion: nn.Linear -> tf.keras.layers.Dense
            self.layer_dict[str(layer_id)] = layers.Dense(32, use_bias=False)
            
        self.output_layer = layers.Dense(128, use_bias=False)

    def call(self, x):
        x = self.tok_embeddings(x) if self.tok_embeddings else x
        for layer in self.layer_dict.values():
            x = layer(x)
        return self.output_layer(x) if self.output_layer else x


def main() -> None:
    # Conversion: torch.distributed.MirroredStrategy replaces init_process_group/device_mesh
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        # Conversion: torch.device("meta") is not needed in TF, we build the graph directly
        model = Transformer()

        # Conversion: device is handled by strategy scope
        device = None 

        pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_transformer(
            model,
            None, # world_mesh
            device,
            parallelize_transformer,
            loss_fn,
        )
        
        # Conversion: m.to_empty is not needed, variables are created on first call or build
        for m in model_parts:
            # m.train() is implicit in TF during training steps, or handled by `training=True`
            pass

        # Conversion: torch.randint -> tf.random.uniform
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
    device: str,
    parallelize_fn,
    loss_fn,
):
    # pp_mesh = world_mesh["pp"] # Simplified for TF
    module_names_per_stage = [
        ["tok_embeddings", "layer_dict.0"],
        ["layer_dict.1", "layer_dict.2"],
        ["layer_dict.3"],
        # Note: 'norm' was in source but not in Transformer class definition, skipping or assuming it was implicit
        ["output_layer"],
    ]

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


def build_pipeline_schedule(stages: list[PipelineStage], loss_fn):
    microbatch_size = 1
    batch_size = 8
    assert batch_size % microbatch_size == 0
    n_microbatches = batch_size // microbatch_size
    # num_total_stages = PP_DEGREE * len(stages) # Logic specific to source
    # assert n_microbatches >= num_total_stages
    schedule_class = get_schedule_class(SCHEDULE_NAME)
    return schedule_class(
        stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


def parallelize_transformer(model: Model, compile=True) -> Model:
    if compile:
        # Conversion: torch.compile -> tf.function
        # Note: In TF, we usually decorate the call method or the function, 
        # but here we wrap the model's call or apply tf.function to the whole object logic.
        # For simplicity in this translation, we assume the model is already efficient 
        # or we apply tf.function to the call method.
        model.call = tf.function(model.call)
        
        # Applying compilation to specific layers (mimicking the loop in source)
        if hasattr(model, 'layer_dict'):
            for layer_id, transformer_block in model.layer_dict.items():
                # In TF, layers are already optimized, but we can re-compile specific logic if needed
                pass 
    return model


def loss_fn(inputs, targets):
    # Conversion: torch.nn.functional.cross_entropy -> tf.keras.losses / tf.nn
    # inputs are logits, targets are class indices
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=inputs,
    )


if __name__ == "__main__":
    main()
```