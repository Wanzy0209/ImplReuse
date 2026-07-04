```python
import tensorflow as tf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Conversion: torch.backends.mps.is_available checks for Apple Silicon GPU.
# TensorFlow generally exposes 'GPU' or 'CPU'. We check for GPU availability.
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

df = sm.datasets.get_rdataset("AirPassengers").data
df = df.shift(1)
df = df.dropna().reset_index(drop=True)

# Conversion: torch.from_numpy -> tf.convert_to_tensor
# Conversion: .reshape(-1,1) -> tf.reshape(..., (-1, 1))
# Conversion: .float() -> dtype=tf.float32
# Conversion: .to(device) -> with tf.device(device): context
with tf.device(device):
    x = tf.convert_to_tensor(df.index.values, dtype=tf.float32)
    x = tf.reshape(x, (-1, 1))

    data = tf.convert_to_tensor(df['value'].values, dtype=tf.float32)

    # Conversion: .unsqueeze(1) -> tf.expand_dims(..., axis=1)
    y = tf.expand_dims(data, axis=1)  # shape (144, 1)

    # Conversion: torch.ones -> tf.ones
    # Conversion: requires_grad=False -> standard tensor (not Variable)
    ones = tf.ones((tf.shape(data)[0], 1), dtype=tf.float32) # shape (144, 1)

    # Conversion: torch.rand -> tf.random.uniform (matches [0, 1) distribution)
    # Conversion: requires_grad=True -> tf.Variable
    w = tf.Variable(tf.random.uniform((2, 1), dtype=tf.float32))

    # Conversion: torch.cat -> tf.concat
    data_aug = tf.concat([x, ones], axis=1) # shape (144, 2)


for epoch in range(100):
    # Conversion: PyTorch autograd -> tf.GradientTape
    with tf.GradientTape() as tape:
        # Conversion: @ operator -> tf.matmul
        pred = tf.matmul(data_aug, w)
        
        # Conversion: torch.abs -> tf.abs
        # Conversion: .mean() -> tf.reduce_mean
        loss = tf.reduce_mean(tf.abs(y - pred))

    # Conversion: loss.backward() -> tape.gradient
    # Note: PyTorch accumulates gradients, so it zeros them. TF GradientTape calculates fresh gradients.
    grads = tape.gradient(loss, w)

    # Conversion: torch.no_grad context is implicit here as we are outside tape
    # Conversion: w.data -= ... -> w.assign_sub(...)
    w.assign_sub(0.01 * grads[0])

    print(f"Epoch {epoch}: loss = {loss.numpy():.4f}")
```