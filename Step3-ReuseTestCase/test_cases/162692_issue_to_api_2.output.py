import tensorflow as tf
import numpy as np

def test_distributed_mean_uneven_sharding():
    """
    Test case for distributed mean calculation with uneven sharding.
    This test mirrors the logic of the PyTorch DTensor.mean bug (Issue 162692)
    but translates it to TensorFlow semantics using tf.distribute.
    
    It leverages tf.keras.initializers.LecunUniform (the similar API) to generate
    the input tensor data.
    """
    
    # Initialize the strategy (MirroredStrategy for local multi-GPU/CPU)
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # 1. Generate the "Truth" tensor using the Similar API (LecunUniform)
    # We use a fixed seed for reproducibility
    initializer = tf.keras.initializers.LecunUniform(seed=42)
    
    # Shape (3, 4) is chosen to ensure uneven sharding with 2 devices (batch size 2)
    # Device 0 gets 2 rows, Device 1 gets 1 row.
    full_shape = (3, 4)
    full_tensor = initializer(shape=full_shape)
    
    # Calculate the expected mean on the full tensor
    expected_mean = tf.reduce_mean(full_tensor)
    print(f"Expected Mean: {expected_mean.numpy()}")

    # 2. Create a dataset and distribute it
    # We slice the tensor to treat rows as individual samples
    dataset = tf.data.Dataset.from_tensor_slices(full_tensor)
    
    # Batch size 2 creates uneven batches for 2 replicas (2 vs 1)
    global_batch_size = 2
    batched_dataset = dataset.batch(global_batch_size, drop_remainder=False)
    
    # Distribute the dataset
    dist_dataset = strategy.experimental_distribute_dataset(batched_dataset)

    # 3. Define the step function to compute mean on each replica
    def step_fn(batch):
        # batch is the local shard for this replica
        # Compute local sum and count
        local_sum = tf.reduce_sum(batch)
        local_count = tf.cast(tf.size(batch), tf.float32)
        return local_sum, local_count

    # 4. Iterate and aggregate
    total_sum = 0.0
    total_count = 0.0

    for batch in dist_dataset:
        # Run the step on all replicas
        per_replica_sum, per_replica_count = strategy.run(step_fn, args=(batch,))
        
        # Aggregate results from all replicas (SUM operation)
        batch_sum = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_sum, axis=None)
        batch_count = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_count, axis=None)
        
        total_sum += batch_sum
        total_count += batch_count

    # 5. Compute final mean
    calculated_mean = total_sum / total_count
    print(f"Calculated Mean
    assert strategy
