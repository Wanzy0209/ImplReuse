```python
import tensorflow as tf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Check for GPU availability (MPS is handled via Metal plugin in TF, mapped to GPU)
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device):
    df = sm.datasets.get_rdataset("AirPassengers").data
    df = df.shift(1)
    df = df.dropna().reset_index(drop=True)

    # Convert index to tensor, reshape, and cast to float32
    x = tf.constant(df.index.values, dtype=tf.float32)
    x = tf.reshape(x, (-1, 1))

    # Convert data to tensor and cast to float32
    data = tf.constant(df['value'].values, dtype=tf.float32)

    # Add dimension to y
    y = tf.expand_dims(data, axis=1)  # shape (144, 1)

    # Create ones tensor
    ones = tf.ones((tf.shape(data)[0], 1), dtype=tf.float32)  # shape (144, 1)

    # Initialize weights as a trainable variable
    w = tf.Variable(tf.random.uniform((2, 1)))

    # Concatenate x and ones
    data_aug = tf.concat([x, ones], axis=1)  # shape (144, 2)

    for epoch in range(100):
        with tf.GradientTape() as tape:
            # Forward pass: Matrix multiplication
            pred = tf.matmul(data_aug, w)

            # Calculate loss: Mean Absolute Error
            loss = tf.reduce_mean(tf.abs(y - pred))

        # Backward pass: Calculate gradients
        gradients = tape.gradient(loss, w)

        # Manual SGD update: w = w - lr * grad
        # In TF, we use assign_sub for in-place update
        w.assign_sub(0.01 * gradients[0])

        print(f"Epoch {epoch}: loss = {loss:.4f}")
```