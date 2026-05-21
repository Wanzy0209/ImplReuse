import torch
"""
Adapted Test Case for TensorFlow API: tf.compat.v1.enable_eager_execution

This test case adapts the PyTorch pipeline parallelism bug reproduction logic
to TensorFlow. The original bug involves `torch.compile` failing with specific
pipeline schedules. Here, we test the TensorFlow equivalent execution mode
control (`tf.compat.v1.enable_eager_execution`) within a distributed context
to verify compatibility.

Run:
python test_tf_eager_distributed.py
"""

import tensorflow as tf
import numpy as np

# --- API Under Test ---
# Explicitly enable eager execution to test compatibility with distributed strategies.
# In TensorFlow 2.x this is default, but we call it to match the API usage pattern.
tf.compat.v1.enable_eager_execution()

# --- Configuration mirroring the PyTorch example ---
PP_DEGREE = 2  # Simulating the parallel degree
BATCH_SIZE = 8
SEQ_LEN = 128  # Reduced from 4096 for minimal test execution speed
VOCAB_SIZE = 128
HIDDEN_DIM = 32

# --- Distributed Setup ---
# Using MirroredStrategy for multi-GPU/single-node distributed training,
# analogous to the local device mesh setup in the PyTorch script.
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')


# --- Model Definition ---
class Transformer(tf.keras.Model):
    """
    Minimal Transformer model mirroring the PyTorch nn.Module structure.
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.tok_embeddings = tf.keras.layers.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        # Create a list of layers to simulate the ModuleDict of layers
        self.layers_list = [tf.keras.layers.Dense(HIDDEN_DIM, use_bias=False) for _ in range(4)]
        self.output_layer = tf.keras.layers.Dense(VOCAB_SIZE, use_bias=False)

    def call(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers_list:
            x = layer(x)
        return self.output_layer(x)


def main() -> None:
    # Initialize the model within the distribution strategy scope
    with strategy.scope():
        model = Transformer()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Generate dummy input data
    # Analogous to torch.randint
    input_ids = tf.random.uniform((BATCH_SIZE, SEQ_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
    labels = input_ids  # Using input_ids as dummy targets for loss calculation

    # --- Execution Step ---
    # This mimics the pipeline schedule step in the original bug report.
    # We verify that the eager execution mode handles the forward/backward pass
    # without crashing, similar to how the bug report checks for failure.
    
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Run the step
    # Note: We wrap in tf.function to test the interaction between eager context
    # and graph compilation, which is a common stress test for TF execution modes.
    # However, the global execution mode remains eager.
    loss_value = train_step(input_ids, labels)

    # --- Assertions ---
    assert loss_value is not None, "Loss calculation failed"
    assert not tf.math.is_nan(loss_value), "Loss is NaN"
    
    print(f"Test Passed. Loss value: {loss_value.numpy()}")
    print("Eager execution is compatible with the distributed model setup.")


if __name__ == "__main__":
    main()