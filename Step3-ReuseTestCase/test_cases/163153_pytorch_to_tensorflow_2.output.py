import torch
import tensorflow as tf
import numpy as np

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """Verification that we have at least 2 gpus to run dist examples"""
    gpus = tf.config.list_physical_devices('GPU')
    return len(gpus) >= min_gpus

# Note: TensorFlow MirroredStrategy handles communication implicitly.
# Explicit module-level prefetching (set_modules_to_forward_prefetch) 
# is a PyTorch FSDP2 specific feature for sharded parameters.
# In MirroredStrategy, parameters are replicated, so this specific 
# API call does not have a direct equivalent.

def main():
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        return

    # Initialize MirroredStrategy
    # This replaces torch.distributed.init_process_group and handles device placement
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Model configuration mirroring the PyTorch example
    vocab_size = 1024
    n_layers = 10
    dim = 4096
    seq_len = 1024
    batch_size = 4

    # Define the model within the strategy scope
    # This replaces the 'with torch.device("meta")' and 'fully_shard' logic
    with strategy.scope():
        inputs = tf.keras.Input(shape=(seq_len,), batch_size=batch_size)
        x = inputs
        
        # Build a simple stack of layers to mimic the Transformer
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
        
        outputs = tf.keras.layers.Dense(vocab_size)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.MeanSquaredError()

    # Inspect model
    model.summary()

    # Prepare dummy data
    x_data = np.random.random((batch_size, seq_len)).astype(np.float32)
    y_data = np.random.random((batch_size, vocab_size)).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
            # Scale loss by number of replicas
            loss = loss / strategy.num_replicas_in_sync
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Run a training step to verify distributed execution
    print("Running training step...")
    for inputs in dist_dataset:
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        print(f"Step completed. Loss: {reduced_loss.numpy()}")
        break

if __name__ == "__main__":
    main()