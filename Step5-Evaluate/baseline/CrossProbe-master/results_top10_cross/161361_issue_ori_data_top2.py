```python
import tensorflow as tf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# PyTorch: device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# TensorFlow: MPS is PyTorch specific, checking for GPU (CUDA) or CPU
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

df = sm.datasets.get_rdataset("AirPassengers").data
df = df.shift(1)
df = df.dropna().reset_index(drop=True)

# PyTorch: x = torch.from_numpy(df.index.values).reshape(-1,1).float().to(device)
with tf.device(device):
    x = tf.convert_to_tensor(df.index.values, dtype=tf.float32)
    x = tf.reshape(x, (-1, 1))

    # PyTorch: data = torch.from_numpy(df['value'].values).float().to(device)
    data = tf.convert_to_tensor(df['value'].values, dtype=tf.float32)
    
    # PyTorch: y = data.float().unsqueeze(1).to(device)
    y = tf.expand_dims(data, axis=1)  # shape (144, 1)
    
    # PyTorch: ones = torch.ones(data.size(0),1,requires_grad=False).to(device)
    ones = tf.ones((tf.shape(data)[0], 1), dtype=tf.float32) # shape (144, 1)
    
    # PyTorch: w = torch.rand(2,1,requires_grad=True).to(device)
    # Note: tf.Variable implies trainable=True by default
    w = tf.Variable(tf.random.uniform((2, 1), dtype=tf.float32))
    
    # PyTorch: data_aug = torch.cat([x, ones], dim=1).to(device)
    data_aug = tf.concat([x, ones], axis=1) # shape (144, 2)


for epoch in range(100):
    # PyTorch: pred = data_aug @ w
    # TensorFlow: Use tf.matmul for matrix multiplication
    with tf.GradientTape() as tape:
        pred = tf.matmul(data_aug, w)
        
        # PyTorch: loss = torch.abs(y - pred).mean()
        loss = tf.reduce_mean(tf.abs(y - pred)) 

    # PyTorch: if w.grad is not None: w.grad.zero_()
    # TensorFlow: Gradients are calculated fresh in each GradientTape context, no manual zeroing needed.

    # PyTorch: loss.backward()
    # TensorFlow: Compute gradients
    grads = tape.gradient(loss, w)

    # PyTorch: with torch.no_grad(): w.data -= 0.01 * w.grad
    # TensorFlow: Update weights. assign_sub is used in-place. 
    # Since this is outside the tape context, it is not tracked for gradients.
    w.assign_sub(0.01 * grads)

    # PyTorch: w.grad.zero_()
    # (Handled by GradientTape lifecycle)

    print(f"Epoch {epoch}: loss = {loss.numpy():.4f}")
```