import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor

class RelayoutModelReplicated(tf.Module):
    """
    Model that relayouts a tensor to a fully replicated layout.
    Analogous to AnyDimsModelEmpty in the PyTorch bug report.
    """
    def __init__(self, mesh):
        super().__init__()
        self.layout = dtensor.Layout.replicated(mesh, rank=2)

    def forward(self, x):
        return dtensor.relayout(x, self.layout)

class RelayoutModelSharded(tf.Module):
    """
    Model that relayouts a tensor to a sharded layout.
    Analogous to AnyDimsModelNull in the PyTorch bug report.
    """
    def __init__(self, mesh):
        super().__init__()
        # Sharding on the second dimension
        self.layout = dtensor.Layout.sharded(mesh, sharding_specs=[dtensor.UNSHARDED, "x"])

    def forward(self, x):
        return dtensor.relayout(x, self.layout)

def process(model, x, name):
    print(f"Processing {name}")
    print(f"Input layout: {x.layout}")

    # Eager mode
    print("Running eager mode...")
    y_eager = model.forward(x)
    print(f"Eager output layout: {y_eager.layout}")

    # Export/Traced mode
    print("Exporting (tracing)...")
    # In TensorFlow, tf.function is the mechanism for tracing/exporting graphs.
    # We wrap the model's forward pass to simulate the export behavior.
    traced_model = tf.function(model.forward)
    y_traced = traced_model(x)
    print(f"Traced output layout: {y_traced.layout}")
    print()

    # Verify correctness to catch potential caching/state pollution bugs
    assert y_eager.layout == model.layout, f"Eager layout mismatch for {name}"
    assert y_traced.layout == model.layout, f"Traced layout mismatch for {name}"

if __name__ == "__main__":
    # Setup Mesh
    mesh = dtensor.create_mesh([("x", 1)], devices=dtensor.all_devices())
    
    # Create input tensor
    t = tf.zeros((4, 4))
    # Initialize with a replicated layout
    d_t = dtensor.DTensor(t, layout=dtensor.Layout.replicated(mesh, rank=2))

    # Process first model (Replicated)
    # This should run fine.
    model_replicated = RelayoutModelReplicated(mesh)
    process(model_replicated, d_t, "ReplicatedModel")

    # Process second model (Sharded)
    # In the PyTorch bug, running a second model with different arguments 
    # immediately after the first resulted in incorrect output shapes due to caching.
    # We test if dtensor.relayout suffers from similar state pollution.
    model_sharded = RelayoutModelSharded(mesh)
    process(model_sharded, d_t, "ShardedModel")