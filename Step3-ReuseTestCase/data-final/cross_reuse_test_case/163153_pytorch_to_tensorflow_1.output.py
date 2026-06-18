import torch
import tensorflow as tf
import tf.experimental.dtensor as dt

def verify_min_tpu_count(min_tpus: int = 2) -> bool:
    """Verification that we have at least 2 TPUs to run dist examples."""
    try:
        # Initialize TPU system
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        # Check device count
        tpu_devices = tf.config.list_logical_devices('TPU')
        return len(tpu_devices) >= min_tpus
    except (ValueError, tf.errors.NotFoundError) as e:
        print(f"TPU initialization failed: {e}")
        return False

def main():
    _min_tpu_count = 2
    if not verify_min_tpu_count(min_tpus=_min_tpu_count):
        print(f"Unable to locate sufficient {_min_tpu_count} TPUs to run this example. Exiting.")
        return

    print("TPU devices initialized successfully.")

    # Original API: torch.distributed.init_process_group(backend=backend, device_id=device)
    # Similar API: tf.experimental.dtensor.create_tpu_mesh(...)
    #
    # In PyTorch, init_process_group sets up the communication backend and process group.
    # In TensorFlow DTensor, create_tpu_mesh defines the physical topology (mesh) of devices
    # used for distributed computation.
    
    num_devices = len(tf.config.list_logical_devices('TPU'))
    
    # Create a mesh. Here we define a 1D mesh for data parallelism (batch dimension).
    # This corresponds to the 'world' setup in PyTorch distributed.
    mesh = dt.create_tpu_mesh(
        mesh_dim_names=['batch'], 
        mesh_shape=[num_devices], 
        mesh_name='global_mesh'
    )
    
    print(f"Created TPU mesh: {mesh}")

    # Define a simple model to shard (similar to Transformer in the PyTorch example)
    class SimpleModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(128, name="dense1")
            self.dense2 = tf.keras.layers.Dense(10, name="dense2")

        def call(self, x):
            return self.dense2(self.dense1(x))

    # Original Logic: fully_shard(layer, **fsdp_kwargs)
    # Adaptation: In DTensor, sharding is handled by Layouts. We define a layout 
    # that replicates the model across the mesh (similar to default FSDP behavior 
    # before specific sharding strategies are applied).
    
    # Define a replicated layout for the model weights
    replicated_layout = dt.Layout([dt.UNSHARDED, dt.UNSHARDED], mesh)

    with dt.context(mesh):
        model = SimpleModel()
        
        # Build the model to initialize variables
        dummy_input = tf.zeros((4, 128))
        _ = model(dummy_input)
        
        # Apply the layout to the model variables (simulating the sharding step)
        # In a real scenario, one might shard specific layers differently.
        for var in model.trainable_variables:
            dt.utils.replicate(var, mesh)

        print("Model variables initialized and sharded on mesh.")
        
        # Verification: Ensure variables are DTensors
        for var in model.trainable_variables:
            assert isinstance(var, dt.DTensor), f"Variable {var.name} is not a DTensor"
            print(f"Variable {var.name} layout: {var.layout}")

if __name__ == "__main__":
    main()