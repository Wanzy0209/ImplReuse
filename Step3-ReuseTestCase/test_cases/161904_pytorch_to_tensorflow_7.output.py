import torch
import tensorflow as tf
import numpy as np

# Define the Transformer model similar to the PyTorch example
class Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = tf.keras.layers.Embedding(128, 32)
        self.layers = [tf.keras.layers.Dense(32, use_bias=False) for _ in range(4)]
        self.output = tf.keras.layers.Dense(128, use_bias=False)

    def call(self, x):
        # Using the similar API: tf.keras.backend.name_scope
        # This mimics the structure where specific parts of the model are scoped
        with tf.keras.backend.name_scope("embeddings"):
            x = self.tok_embeddings(x)

        with tf.keras.backend.name_scope("transformer_layers"):
            for layer in self.layers:
                x = layer(x)

        with tf.keras.backend.name_scope("output"):
            x = self.output(x)
        return x

def main():
    # Setup distributed environment (MirroredStrategy for single-node multi-gpu)
    # This corresponds to the torch.distributed.init_process_group and device mesh setup
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        # Initialize model
        model = Transformer()

        # Optimizer and Loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Mimic torch.compile using tf.function
        # This is the TensorFlow equivalent of compiling the model for performance
        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                # We can also use name_scope here to verify it works in the compiled graph
                with tf.keras.backend.name_scope("forward_pass"):
                    predictions = model(inputs, training=True)
                    loss = loss_fn(targets, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        # Create dummy data
        batch_size = 8
        seq_len = 4096
        # Note: TF Embedding expects int inputs
        input_ids = tf.random.uniform((batch_size, seq_len), minval=0, maxval=128, dtype=tf.int32)
        # Create dummy targets (float for MSE)
        labels = tf.random.uniform((batch_size, seq_len, 128), dtype=tf.float32)

        # Run the step
        print("Running compiled step with name_scope...")
        loss = train_step(input_ids, labels)
        print(f"Step completed. Loss: {loss.numpy()}")

        # Verify that the name_scope actually affected the graph structure
        # by checking variable names (optional but good for verification)
        print("Variable names in model:")
        for var in model.trainable_variables:
            print(f"- {var.name}")

if __name__ == "__main__":
    main()