```python
import tensorflow as tf

# PyTorch's device="cuda" maps to '/GPU:0' context in TensorFlow
with tf.device('/GPU:0'):
    # PyTorch's .item() is equivalent to .numpy().item() in TensorFlow
    print(tf.ones(1).numpy().item())
```