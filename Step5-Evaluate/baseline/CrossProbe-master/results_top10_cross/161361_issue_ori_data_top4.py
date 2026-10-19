```python
import tensorflow as tf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# TensorFlow handles device placement automatically (GPU/MPS if available, else CPU)
# No explicit device object creation or tensor movement needed like in PyTorch
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

df = sm.datasets.get_rdataset("AirPassengers").data
df = df.shift(1)
df = df.dropna().reset_index(drop=True)

# PyTorch: torch.from_numpy(df.index.values).reshape(-1,1).float().to(device)
# TF: tf.convert_to_tensor, tf.reshape, tf.cast
x = tf.cast(tf.reshape(df.index.values, (-1, 1)), tf.float32)
data = tf.cast(tf.convert_to_tensor(df['value'].values), tf.float32)

# PyTorch: data.float().unsqueeze(1).to(device)
# TF: tf.expand_dims(..., axis=1)
y = tf.expand_dims(data, axis=1)  # shape (144, 1)

# PyTorch: torch.ones(data.size(0),1,requires_grad=False).to(device)
# TF: tf.ones (gradients are not tracked for constants)
ones = tf.ones((tf.shape(data)[0], 1), dtype=tf.float32) # shape (144, 1)

# PyTorch: torch.rand(2,1,requires_grad=True).to(device)
# TF: tf.Variable with tf.random.uniform
w = tf.Variable(tf.random.uniform((2, 1)), dtype=tf.float32)

# PyTorch: torch.cat([x, ones], dim=1).to(device)
# TF: tf.concat([x, ones], axis=1)
data_aug = tf.concat([x, ones], axis=1) # shape (144, 2)


for epoch in range(100):
    # PyTorch: pred = data_aug @ w
    # TF: tf.matmul
    # PyTorch: loss = torch.abs(y - pred).mean()
    # TF: tf.reduce_mean(tf.abs(...))
    # Note: In TF, we use tf.GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        pred = tf.matmul(data_aug, w)
        loss = tf.reduce_mean(tf.abs(y - pred)) 

    # PyTorch: if w.grad is not None: w.grad.zero_()
    # Not needed in TF as gradients are not accumulated across tapes

    # PyTorch: loss.backward()
    # TF: tape.gradient
    grads = tape.gradient(loss, w)

    # PyTorch: with torch.no_grad(): w.data -= 0.01 * w.grad
    # TF: w.assign_sub(...) (performed outside the tape context to avoid differentiating the update)
    w.assign_sub(0.01 * grads)  

    # PyTorch: w.grad.zero_()
    # Not needed in TF

    print(f"Epoch {epoch}: loss = {loss.numpy():.4f}")
```