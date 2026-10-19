```python
import tensorflow as tf

# torch.ones creates a tensor of ones; device placement handled via context
with tf.device('/GPU:0'):
    # .item() extracts the scalar value from the tensor
    print(tf.ones(1).numpy().item())
```