import torch
import tensorflow as tf
import numpy as np

# Replicating the structure of the original bug report to test tf.linspace
# in a compiled context (tf.function) within a loop.

BATCH_SIZE = 64
CHANNELS, IMG_SIZE = 3, 224
GRID_SIZE = 13
NUM_POINTS = 5000  # Corresponds to the expansion size in the original bug

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimicking the Linear layer from the original code
        # Input size: NUM_POINTS * GRID_SIZE * GRID_SIZE * CHANNELS
        self.fc = tf.keras.layers.Dense(16)

    @tf.function  # Enabling compilation (analogous to torch.compile)
    def call(self, x):
        per_channel = []
        for i in range(CHANNELS):
            # Original Bug: channel = x[:,i,...].expand(5000,-1,-1,-1)
            # The bug was that expand (view) was treated as repeat (copy), causing OOM.
            
            # Adaptation: Use tf.linspace to generate a tensor inside the loop.
            # We use tf.linspace to generate a coordinate vector of size NUM_POINTS.
            # This tests if tf.linspace handles memory correctly in a compiled loop,
            # similar to the expand/repeat issue.
            coords = tf.linspace(0.0, 1.0, NUM_POINTS)
            coords = tf.reshape(coords, (NUM_POINTS, 1, 1, 1))

            # Extract the channel slice
            channel = x[:, i, :, :] # Shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE)

            # Mimic the expand behavior using broadcasting
            # We broadcast the channel to (NUM_POINTS, BATCH_SIZE, IMG_SIZE, IMG_SIZE)
            expanded_channel = tf.broadcast_to(channel, (NUM_POINTS, BATCH_SIZE, IMG_SIZE, IMG_SIZE))

            # Combine the linspace coords with the expanded channel
            # (Simulating processing similar to grid_sample)
            patch = expanded_channel * coords

            # Mimic the dimensionality reduction of grid_sample (output size GRID_SIZE)
            # using resize to keep tensor sizes manageable and logic similar.
            patch = tf.image.resize(patch, [GRID_SIZE, GRID_SIZE]) # Shape (NUM_POINTS, BATCH_SIZE, GRID_SIZE, GRID_SIZE)

            # Mimic transpose and flatten from original code
            patch = tf.transpose(patch, [1, 0, 2, 3]) # Shape (BATCH_SIZE, NUM_POINTS, GRID_SIZE, GRID_SIZE)
            patch = tf.reshape(patch, [BATCH_SIZE, NUM_POINTS, -1]) # Shape (BATCH_SIZE, NUM_POINTS, GRID_SIZE*GRID_SIZE)

            per_channel.append(patch)

        # Concatenate along the feature dimension
        x = tf.concat(per_channel, axis=2) # Shape (BATCH_SIZE, NUM_POINTS, GRID_SIZE*GRID_SIZE*CHANNELS)
        
        # Flatten for the Dense layer
        x = tf.reshape(x, [BATCH_SIZE, -1])
        
        # Pass through the linear layer
        x = self.fc(x)
        return x

def main():
    model = MyModel()

    # Run multiple iterations to check for memory leaks or allocation issues
    # similar to the original bug report's loop.
    for _ in range(10):
        x = tf.random.normal((BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x)
            loss = tf.reduce_sum(y)
        
        grads = tape.gradient(loss, x)
        
        # Assertion to ensure computation happened correctly
        assert grads is not None
        print(f"Iteration complete. Output shape: {y.shape}, Loss: {loss.numpy()}")

if __name__ == "__main__":
    main()