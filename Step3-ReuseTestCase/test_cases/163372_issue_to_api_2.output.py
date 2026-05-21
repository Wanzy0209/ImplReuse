import torch
import tensorflow as tf
import numpy as np

# Replicating the structure of the PyTorch bug report to test the similar API
# (tf.experimental.numpy.around) in a compiled context with loops.

BATCH_SIZE = 64
CHANNELS, IMG_SIZE = 3, 224
GRID_SIZE = 13

class MyModel(tf.Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters similar to the PyTorch model
        self.grid = tf.Variable(tf.random.normal((5000, GRID_SIZE, GRID_SIZE, 2)), name='grid')
        self.w = tf.Variable(tf.random.normal((GRID_SIZE*GRID_SIZE*CHANNELS, 16)), name='w')
        self.b = tf.Variable(tf.zeros((16,)), name='b')

    @tf.function # Equivalent to torch.compile
    def __call__(self, x):
        per_channel = []
        for i in range(CHANNELS):
            # Original PyTorch code used expand() which was misinterpreted as repeat().
            # Here we use tf.broadcast_to (semantic equivalent of expand) and then
            # apply the similar API: tf.experimental.numpy.around.
            
            # Slice the channel
            channel = x[:, i, ...]
            
            # Expand dimensions (broadcasting)
            # Note: In PyTorch, expand(5000, -1, -1, -1) on a (Batch, H, W) tensor 
            # results in (5000, Batch, H, W) if the dim matches, or (Batch, 5000, H, W) 
            # depending on input. The original code: x is (B, C, H, W). x[:, i] is (B, H, W).
            # expand(5000, -1, -1, -1) makes it (5000, B, H, W).
            channel_expanded = tf.broadcast_to(channel, [5000, BATCH_SIZE, IMG_SIZE, IMG_SIZE])
            
            # Apply the similar API: tf.experimental.numpy.around
            # This tests if the API behaves correctly inside a compiled loop,
            # analogous to the failing expand() call.
            processed_channel = tf.experimental.numpy.around(channel_expanded, decimals=4)
            
            # Mimic the rest of the pipeline (simplified for the test case)
            # Original: grid_sample -> transpose -> flatten
            # We flatten to match dimensions for the linear layer
            # Shape: (5000, BATCH_SIZE, IMG_SIZE, IMG_SIZE) -> (5000, BATCH_SIZE * IMG_SIZE * IMG_SIZE)
            # But wait, the original code transposes (0,1) first.
            # Original: patch = patch.transpose(0,1).flatten(start_dim=2)
            # Let's just flatten to ensure we can feed it to the linear layer logic roughly.
            patch = tf.reshape(processed_channel, [5000, -1])
            per_channel.append(patch)
        
        # Concatenate along the feature dimension
        x = tf.concat(per_channel, axis=1)
        
        # Apply linear layer
        x = tf.matmul(x, self.w) + self.b
        return x

def main():
    model = MyModel()
    
    # Run the model to trigger compilation (tracing)
    x = tf.random.normal((BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE))
    
    # Run a few iterations to ensure stability
    for _ in range(2):
        y = model(x)
    
    # Basic assertion to ensure the model runs and produces output of correct shape
    # The original bug caused OOM. If this runs, it's a good sign for the TF API.
    assert y.shape == (BATCH_SIZE, 16)
    print("Test passed. tf.experimental.numpy.around works within compiled loop.")

if __name__ == "__main__":
    main()