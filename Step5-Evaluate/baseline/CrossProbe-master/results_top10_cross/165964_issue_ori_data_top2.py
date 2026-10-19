```python
import tensorflow as tf

# torch.ones(1, device="cuda") creates a tensor of ones on the GPU.
# In TensorFlow, device placement is handled via the tf.device context manager.
# .item() extracts the value as a standard Python scalar.
with tf.device('/GPU:0'):
    print(tf.ones(1).numpy().item())
```