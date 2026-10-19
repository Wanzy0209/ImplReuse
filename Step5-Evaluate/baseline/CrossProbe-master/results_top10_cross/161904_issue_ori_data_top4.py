```python
"""
Run:

python repro.py
"""

import tensorflow as tf
import numpy as np
from typing import List, Optional, Tuple, Union

# Conversion: torch.distributed.device_mesh -> tf.distribute.Strategy
# Conversion: torch.distributed.pipelining -> Custom implementation for TF
# Conversion: torchtitan -> Custom implementation for TF

PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"


class Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conversion: torch.nn.Embedding -> tf.keras.layers.Embedding
        self.tok_embeddings = tf.keras.layers.Embedding(128, 32, name="tok_embeddings")
        
        # Conversion: torch.nn.ModuleDict -> Python dict of layers
        self.layers = {}
        for layer_id in range(4):
            # Conversion: torch.nn.Linear -> tf.keras.layers.Dense
            self.layers[str(layer_id)] = tf.keras.layers.Dense(32, use_bias=False, name=f"layers.{layer_id}")
            
        self.output_layer = tf.keras.layers.Dense(128, use_bias=False, name="output")

    def call(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers.values():
            x = layer(x)
        return self.output_layer(x)


# Mocking PipelineStage for structure preservation
class PipelineStage:
    def __init__(self, submod, is_first=False, is_last=False):
        self.submod = submod
        self.is_first = is_first
        self.is_last = is_last


# Mocking Schedule for structure preservation
class Schedule:
    def __init__(self, stages, n_microbatches, loss_fn):
        self.stages = stages
        self.n_microbatches = n_microbatches
        self.loss_fn = loss_fn

    def step(self, *args, **kwargs):
        # Simplified execution logic for demonstration
        # In a real TF pipeline parallelism setup, this would involve tf.function and complex device placement
        input_tensor = args[0] if args else kwargs.get('input')
        target = kwargs.get('target')
        losses = kwargs.get('losses')
        
        x = input_tensor
        for stage in self.stages:
            x = stage.submod(x)
            
        if target is not None and self.loss_fn:
            loss = self.loss_fn(target, x)
            if losses is not None:
                losses.append(loss)


def pipeline_module_split(model, pp_mesh, schedule_name, device, module_names_per_stage):
    """
    Custom implementation to mimic torchtitan.distributed.pipeline_parallel.pipeline_module_split
    Splits the Keras model into sub-models based on layer names.
    """
    stages = []
    model_parts = []
    
    # Helper to get layer by name from the main model
    def get_layer(name):
        if name == "tok_embeddings":
            return model.tok_embeddings
        elif name == "output":
            return model.output_layer
        elif name.startswith("layers."):
            return model.layers[name.split(".")[1]]
        # Note: 'norm' was in source but not defined in Transformer class, skipping or assuming identity
        return tf.keras.layers.Lambda(lambda x: x) 

    for names in module_names_per_stage:
        # Create a sub-model (or sequential container) for the stage
        # In TF, we can wrap these in a Model or just a list. 
        # Here we create a simple Model to act as the 'submod'
        inputs = tf.keras.Input(shape=(None, 32)) # Assuming intermediate dim is 32
        temp = inputs
        layers_in_stage = []
        
        for name in names:
            layer = get_layer(name)
            layers_in_stage.append(layer)
            # We can't easily build a static subgraph without knowing input shapes perfectly
            # so we will store the layers and wrap them in a callable Model
            pass
            
        # Construct a functional model for the stage
        # This is a simplification. Real splitting requires tracing.
        # We will create a dummy model that holds these layers.
        class SubModel(tf.keras.Model):
            def __init__(self, layers_list):
                super().__init__()
                self.layers_list = layers_list
            
            def call(self, x):
                for l in self.layers_list:
                    x = l(x)
                return x
                
        submod = SubModel(layers_in_stage)
        model_parts.append(submod)
        
        # Determine stage position (simplified)
        is_first = (names == module_names_per_stage[0])
        is_last = (names == module_names_per_stage[-1])
        
        stages.append(PipelineStage(submod, is_first, is_last))
        
    return stages, model_parts


def main() -> None:
    # Conversion: torch.distributed.get_node_local_rank -> handled by TF Strategy
    # Conversion: torch.device -> handled by TF Strategy
    # Conversion: torch.distributed.init_process_group -> tf.distribute.MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        # Conversion: init_device_mesh -> Strategy scope handles device mesh
        # mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("pp",))
        
        # Conversion: with torch.device("meta") -> TF builds graph first, no explicit meta device needed
        model = Transformer()

        # Note: pipeline_transformer logic is adapted for TF
        pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_transformer(
            model,
            strategy, # Passing strategy instead of mesh
            None,     # Device handled by strategy
            parallelize_transformer,
            loss_fn,
        )
        
        # Conversion: m.to_empty -> Not needed in TF, weights initialized in scope
        for m in model_parts:
            # m.train() is default behavior for layers in TF unless explicitly set to inference
            pass

        # Conversion: torch.randint -> tf.random.uniform
        input_ids = tf.cast(tf.random.uniform((8, 4096), minval=0, maxval=128, dtype=tf.int32), tf.int32)
        labels = input_ids

        losses: Union[List[tf.Tensor], None]
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
    model: tf.keras.Model,
    strategy, # Replaced world_mesh
    device,   # Not used in TF, handled by strategy
    parallelize_fn,
    loss_fn,
):
    # pp_mesh = world_mesh["pp"] -> Strategy handles distribution
    module_names_per_stage = [
        ["tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3"],
        # ["norm"], # 'norm' was in source but not in Transformer class
        ["output"],
    ]

    stages, model_parts = pipeline_module_split(
        model,
        strategy, # Passing strategy
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


def build_pipeline_schedule(stages: List[PipelineStage], loss_fn):
    microbatch_size = 1
    batch_size = 8
    assert batch_size % microbatch_size == 0
    n_microbatches = batch_size // microbatch_size
    num_total_stages = PP_DEGREE * len(stages)
    assert n_microbatches >= num_total_stages
    
    # Conversion: get_schedule_class -> Return our custom Schedule class
    # In TF, schedules are often implicit in the data pipeline or tf.function loops
    return Schedule(
        stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


def parallelize_transformer(model: tf.keras.Model, compile=True) -> tf.keras.Model:
    if compile:
        # Conversion: torch.compile -> tf.function (applied to call method or layers)
        # Note: tf.config.optimizer.set_jit(True) is the global equivalent of torch.compile mode
        # Here we apply tf.function to the call method
        model.call = tf.function(model.call)
        
        # Applying compile to specific layers as in source
        for layer_id, transformer_block in model.layers.items():
             # Assuming layers in the dict are the blocks
             if hasattr(transformer_block, 'call'):
                 transformer_block.call = tf.function(transformer_block.call)
                 
    return model


def loss_fn(targets, inputs):
    # Conversion: torch.nn.functional.cross_entropy -> tf.keras.losses.SparseCategoricalCrossentropy
    # Note: TF loss functions typically take (y_true, y_pred)
    # Source: inputs (logits), targets (labels)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Reshape logic
    inputs_shape = tf.shape(inputs)
    inputs_flat = tf.reshape(inputs, [-1, inputs_shape[-1]])
    targets_flat = tf.reshape(targets, [-1])
    
    return cce(targets_flat, inputs_flat)


if __name__ == "__main__":
    main()
```