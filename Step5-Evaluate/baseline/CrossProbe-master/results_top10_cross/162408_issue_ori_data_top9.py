```python
import tensorflow as tf

# torch.set_default_dtype(torch.bfloat16)
# Conversion: TensorFlow does not have a global default dtype setting for tensor creation.
# Types are explicit. Since the following code uses int32, this is ignored.

device = "/CPU:0" # Conversion: PyTorch "cpu" maps to TensorFlow "/CPU:0"
# device="/GPU:0" # Conversion: PyTorch "cuda" maps to TensorFlow "/GPU:0"

with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)
    
    # Conversion: torch.cat((zeros, cumsum)) -> tf.concat([zeros, cumsum], axis=0)
    # Conversion: torch.zeros -> tf.zeros
    # Conversion: torch.cumsum(..., dim=0) -> tf.cumsum(..., axis=0)
    cu_seqlens = tf.concat([tf.zeros([1], dtype=tf.int32), tf.cumsum(seqlens, axis=0)], axis=0)

# Ground truth answer: [   0,  434,  512,  613, 1024]
# Conversion: Access .numpy() to print values in eager execution mode
print(f"cu_seqlens={cu_seqlens.numpy()}")
```