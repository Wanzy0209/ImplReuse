```python
import tensorflow as tf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Conversion: Check for GPU availability (MPS is PyTorch specific, TF uses GPU/CPU)
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    df = sm.datasets.get_rdataset("AirPassengers").data
    df = df.shift(1)
    df = df.dropna().reset_index(drop=True)

    # Conversion: torch.from_numpy -> tf.constant, .reshape -> .reshape
    x = tf.constant(df.index.values.reshape(-1, 1), dtype=tf.float32)
    data = tf.constant(df['value'].values, dtype=tf.float32)
    
    # Conversion: .unsqueeze(1) -> tf.expand_dims(..., axis=1)
    y = tf.expand_dims(data, axis=1)  # shape (144, 1)
    
    # Conversion: torch.ones -> tf.ones
    ones = tf.ones((tf.shape(data)[0], 1), dtype=tf.float32) # shape (144, 1)
    
    # Conversion: torch.rand -> tf.random.uniform, requires_grad=True -> tf.Variable
    w = tf.Variable(tf.random.uniform((2, 1)), dtype=tf.float32)
    
    # Conversion: torch.cat -> tf.concat
    data_aug = tf.concat([x, ones], axis=1) # shape (144, 2)

    for epoch in range(100):
        # Conversion: PyTorch autograd -> tf.GradientTape
        with tf.GradientTape() as tape:
            pred = data_aug @ w
            # Conversion: torch.abs(...).mean() -> tf.reduce_mean(tf.abs(...))
            loss = tf.reduce_mean(tf.abs(y - pred))

        # Conversion: loss.backward() -> tape.gradient
        gradients = tape.gradient(loss, w)

        # Conversion: Manual weight update (w.data -= lr * w.grad) -> w.assign_sub(lr * grad)
        # Note: TF operations outside GradientTape are effectively 'no_grad'
        w.assign_sub(0.01 * gradients[0])

        # Conversion: loss.item() -> loss.numpy()
        print(f"Epoch {epoch}: loss = {loss.numpy():.4f}")
```