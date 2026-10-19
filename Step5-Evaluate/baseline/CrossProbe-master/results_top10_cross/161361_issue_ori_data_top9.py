```python
import tensorflow as tf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Conversion: PyTorch device selection to TensorFlow device availability check
# Note: TensorFlow handles device placement automatically, but we check for GPU availability
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

df = sm.datasets.get_rdataset("AirPassengers").data
df = df.shift(1)
df = df.dropna().reset_index(drop=True)

# Conversion: torch.from_numpy -> tf.constant
# Conversion: .reshape(-1,1) -> reshape(-1, 1)
# Conversion: .float() -> dtype=tf.float32
# Conversion: .to(device) -> handled by TF context or automatic placement
with tf.device(device_name):
    x = tf.constant(df.index.values.reshape(-1, 1), dtype=tf.float32)
    data = tf.constant(df['value'].values, dtype=tf.float32)
    
    # Conversion: unsqueeze(1) -> tf.expand_dims(..., axis=1)
    y = tf.expand_dims(data, axis=1)  # shape (144, 1)
    
    # Conversion: torch.ones -> tf.ones
    ones = tf.ones((tf.shape(data)[0], 1), dtype=tf.float32) # shape (144, 1)
    
    # Conversion: torch.rand -> tf.random.uniform
    # Conversion: requires_grad=True -> tf.Variable
    w = tf.Variable(tf.random.uniform((2, 1), dtype=tf.float32))
    
    # Conversion: torch.cat -> tf.concat
    data_aug = tf.concat([x, ones], axis=1) # shape (144, 2)


for epoch in range(100):
    # Conversion: PyTorch autograd to TensorFlow GradientTape
    with tf.GradientTape() as tape:
        # Conversion: @ operator -> tf.matmul
        pred = tf.matmul(data_aug, w)
        
        # Conversion: torch.abs(...).mean() -> tf.reduce_mean(tf.abs(...))
        loss = tf.reduce_mean(tf.abs(y - pred))

    # Conversion: loss.backward() -> tape.gradient
    grads = tape.gradient(loss, w)

    # Conversion: Manual weight update
    # Note: PyTorch uses torch.no_grad() context here. 
    # In TensorFlow, assign_sub does not create a gradient tape operation, so no context is needed.
    # Note: PyTorch zeros grad here. TF does not accumulate gradients in the variable by default.
    w.assign_sub(0.01 * grads)

    # Conversion: loss.item() -> loss.numpy()
    print(f"Epoch {epoch}: loss = {loss.numpy():.4f}")
```