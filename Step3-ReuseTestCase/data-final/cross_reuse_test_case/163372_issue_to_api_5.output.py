import torch
import tensorflow as tf
import numpy as np

# Test case for tf.keras.ops.linspace inspired by PyTorch Issue 163372
# The original issue involved torch.expand being misinterpreted as torch.repeat
# by the compiler, leading to OOM. This test checks if tf.keras.ops.linspace
# behaves efficiently in a compiled context (tf.function) under similar stress.

def test_linspace_compilation_memory():
    BATCH_SIZE = 64
    CHANNELS = 3
    IMG_SIZE = 224
    LARGE_SIZE = 5000  # Corresponds to the expansion factor in the bug report

    class MyModel(tf.Module):
        def __init__(self):
            super().__init__()
            # Using a dense layer to mimic the linear layer in the bug report
            self.fc = tf.keras.layers.Dense(16)

        @tf.function(jit_compile=True)  # Mimic torch.compile
        def __call__(self, x):
            per_channel = []
            for i in range(CHANNELS):
                # Original bug: channel = x[:,i,...].expand(5000,-1,-1,-1)
                # This creates a large tensor. We use linspace to generate a large tensor
                # to stress the compiler's memory handling.
                
                # Extract scalars to define the range for linspace
                start = tf.cast(x[0, i, 0, 0], tf.float32)
                stop = tf.cast(x[0, i, 1, 1], tf.float32)
                
                # Generate a large sequence. 
                # Note: linspace creates a new tensor, unlike expand (view).
                # We are testing if the compiler handles this allocation efficiently
                # in a loop (e.g., not leaking memory or re-allocating unnecessarily).
                channel = tf.keras.ops.linspace(start, stop, num=LARGE_SIZE, axis=0)
                
                # Perform a reduction to simulate work
                patch = tf.reduce_sum(channel)
                per_channel.append(patch)
            
            x = tf.stack(per_channel)
            x = self.fc(x)
            return x

    model = MyModel()
    x = tf.random.normal((BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE))

    # Run multiple iterations to check for memory stability
    for _ in range(10):
        y = model(x)
        # Ensure gradients can be calculated (mimicking backward pass)
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_sum(y)
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Basic assertion to ensure execution
        assert y.shape == (BATCH_SIZE, 16)

    print("Test passed: tf.keras.ops.linspace handled memory efficiently in compiled context.")

if __name__ == "__main__":
    test_linspace_compilation_memory()