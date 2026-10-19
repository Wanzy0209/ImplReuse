```python
import tensorflow as tf

# torch.ones(1, device="cuda") creates a tensor of ones on the GPU
# In TensorFlow, device placement is typically handled via a context manager
with tf.device('/GPU:0'):
    # .item() extracts the value as a Python scalar
    # In TF 2.x (eager execution), we use .numpy().item() to get the value
    print(tf.ones([1]).numpy().item())
```