```python
"""
Run:

TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:12346"]}, "task": {"type": "worker", "index": 0}}' python repro.py
TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:12346"]}, "task": {"type": "worker", "index": 1}}' python repro.py
"""

import tensorflow as tf
from typing import List, Tuple, Optional, Any, Callable

# Conversion: Constants remain the same
PP_DEGREE = 2
SCHEDULE_NAME = "ZBVZeroBubble"  # or "DualPipeV"


# Conversion: torch.nn.Module -> tf.keras.Model
class Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = tf.keras.layers.Embedding(128, 32)
        self.layers = {}
        for layer_id in range(4):
            self.layers[str(layer_id)] = tf.keras.layers.Dense(32, use_bias=False)
        # Conversion: Added norm to match the module_names_per_stage in the source
        self.norm = tf.keras.layers.LayerNormalization()
        self.output = tf.keras.layers.Dense(128, use_bias=False)

    def call(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers.values():
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


# Conversion: Mocking torch.distributed.pipelining.PipelineStage
class PipelineStage:
    def __init__(self, submod, stage_id, num_stages, rank):
        self.submod = submod
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.rank = rank
        # Conversion: Logic to determine first/last stage based on rank assignment
        # Assuming round-robin assignment of stages to ranks for this translation
        self.is_first = (stage_id == 0)
        self.is_last = (stage_id == num_stages - 1)


# Conversion: Mocking torchtitan.distributed.pipeline_parallel.pipeline_module_split
def pipeline_module_split(
    model: Transformer,
    pp_mesh,  # In TF, this is conceptually the strategy
    schedule_name: str,
    device: tf.device,
    module_names_per_stage: List[List[str]],
):
    stages = []
    model_parts = []

    # Create sub-models for each stage
    for i, names in enumerate(module_names_per_stage):
        layers = []
        for name in names:
            if '.' in name:
                # Handle nested access like 'layers.0'
                parts = name.split('.')
                obj = model
                for p in parts:
                    if p.isdigit():
                        obj = obj.layers[int(p)]
                    else:
                        obj = getattr(obj, p)
                layers.append(obj)
            else:
                layers.append(getattr(model, name))
        
        # Create a Keras Model for this stage
        # Note: This is a simplified construction. Real pipeline partitioning requires functional API slicing.
        inputs = tf.keras.Input(shape=(None, 32), batch_size=8) # Placeholder shape
        x = inputs
        for layer in layers:
            # Check if layer is Embedding (input shape differs)
            if isinstance(layer, tf.keras.layers.Embedding):
                 # Re-defining input for embedding stage
                 inputs = tf.keras.Input(shape=(None,), batch_size=8, dtype=tf.int32)
                 x = layer(inputs)
            else:
                 x = layer(x)
        
        stage_model = tf.keras.Model(inputs=inputs, outputs=x)
        model_parts.append(stage_model)

    # Assign stages to ranks (simplified logic for translation)
    # In a real TF pipeline, this is handled by the Strategy.
    # Here we create PipelineStage objects for all, but the Schedule will filter by rank.
    num_stages = len(module_names_per_stage)
    for i, m in enumerate(model_parts):
        # We pass the global rank here, though in TF this is usually implicit
        rank = i % PP_DEGREE 
        stages.append(PipelineStage(m, i, num_stages, rank))

    return stages, model_parts


# Conversion: Mocking torch.distributed.pipelining.schedules.get_schedule_class
def get_schedule_class(name: str):
    class Schedule:
        def __init__(self, stages, n_microbatches, loss_fn):
            self.stages = stages
            self.n_microbatches = n_microbatches
            self.loss_fn = loss_fn
            # Identify which stage belongs to the current worker
            # In TF, we use the resolver to find the current task ID
            resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
            current_rank = resolver.task_id if resolver.task_id is not None else 0
            
            # Filter stages for this rank (assuming round-robin mapping)
            self.my_stages = [s for s in stages if s.rank == current_rank]

        def step(self, args=None, target=None, losses=None):
            # Conversion: Implementing the step logic
            # In a real pipeline, this would involve send/recv ops.
            # Here we execute the local stages sequentially.
            
            x = args
            for stage in self.my_stages:
                if stage.is_first and x is not None:
                    x = stage.submod(x)
                elif not stage.is_first:
                    # In a real distributed setting, x would be received from previous rank
                    # For this translation, we pass a dummy tensor if no input is available
                    # to maintain structure validity.
                    if x is None:
                        x = tf.zeros((8, 4096, 32)) # Dummy input
                    x = stage.submod(x)
                
                if stage.is_last and target is not None:
                    loss = self.loss_fn(x, target)
                    if losses is not None:
                        losses.append(loss)
            
            return x, losses

    return Schedule


def main() -> None:
    # Conversion: torch.distributed.get_node_local_rank -> TF Cluster Resolver
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # Conversion: torch.device -> Strategy Scope
    with strategy.scope():
        # Conversion: torch.distributed.init_process_group is handled by TF_CONFIG and Strategy init
        
        # Conversion: init_device_mesh is implicit in MultiWorkerMirroredStrategy
        # We use the strategy's scope to distribute variables.
        
        # Conversion: torch.device("meta") -> Build model in strategy scope
        # TF doesn't have a direct "meta" device, but we build the graph first.
        model = Transformer()

        # Conversion: pipeline_transformer call
        # We pass the strategy as the mesh equivalent
        pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_transformer(
            model,
            strategy, # world_mesh equivalent
            None,     # device equivalent (handled by strategy)
            parallelize_transformer,
            loss_fn,
        )
        
        # Conversion: m.to_empty -> Not needed in TF/Keras, weights are initialized in scope
        # m.train() -> Keras models are trainable by default unless set otherwise

        # Conversion: torch.randint -> tf.random.uniform
        input_ids = tf.random.uniform((8, 4096), minval=0, maxval=128, dtype=tf.int32)
        labels = input_ids

        losses: List[tf.Tensor] = []
        targets, losses_list = (labels, losses) if has_last_stage else (None, None)
        
        if has_first_stage:
            pp_schedule.step(
                args=input_ids,
                target=targets,
                losses=losses_list,
            )
        else:
            pp_schedule.step(
                target=targets,
                losses=losses_list,
            )


def pipeline_transformer(
    model: tf.keras.Model,
    strategy, # world_mesh
    device,   # Not used in TF, handled by strategy
    parallelize_fn,
    loss_fn,
):
    # Conversion: pp_mesh = world_mesh["pp"]
    # In this simplified translation, we treat the whole strategy as the mesh
    
    module_names_per_stage = [
        ["tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3"],
        ["norm", "output"],
    ]

    stages, model_parts = pipeline_module_split(
        model,
        strategy,
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
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    current_rank = resolver.task_id if resolver.task_id is not None else 0

    for stage in stages:
        # Check if this stage belongs to this worker
        if stage.rank == current_rank:
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
    # num_total_stages logic omitted for simplicity as it depends on specific mesh config
    schedule_class = get_schedule_class(SCHEDULE_NAME)
    return schedule_class(
        stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


def parallelize_transformer(model: tf.keras.Model, compile=True) -> tf.keras.Model:
    if compile:
        # Conversion: torch.compile -> tf.function (jit_compile)
        # Note: Applying tf.function to a Keras Model usually happens during call or via model.compile
        # Here we apply it to the call method or specific layers if needed.
        # For simplicity, we just return the model, as Keras handles graph compilation.
        # To strictly mimic the source:
        for layer_id, transformer_block in model.layers.items():
            # If it's a layer, we can wrap it, but Keras layers are already optimized.
            # We assume the user wants XLA compilation.
            model.layers[layer_id] = tf.function(transformer_block, jit_compile=True)
    return model


def loss_fn(inputs, targets):
    # Conversion: torch.nn.functional.cross_entropy -> tf.keras.losses
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # View(-1, inputs.size(-1)) is equivalent to flattening the batch and sequence dims
    loss = scce(targets, inputs)
    return tf.reduce_mean(loss)


if __name__ == "__main__":
    main()
```