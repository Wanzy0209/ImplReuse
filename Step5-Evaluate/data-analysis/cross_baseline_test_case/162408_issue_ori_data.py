```python
import tensorflow as tf

# Conversion: PyTorch sets default dtype globally. In TF, this is typically handled via Keras mixed precision policy.
# Since the following logic uses explicit int32 types, this setting is commented out to avoid type mismatch errors.
# torch.set_default_dtype(torch.bfloat16)

device = "/cpu:0"
# device="/gpu:0"

# Conversion: In TF, device placement is handled via a context manager.
with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

    # Conversion: torch.zeros -> tf.zeros (Explicitly int32 to match cumsum output for concat)
    # Conversion: torch.cumsum -> tf.cumsum (dim -> axis)
    # Conversion: torch.cat -> tf.concat
    # Conversion: .to() -> tf.cast
    cu_seqlens = tf.cast(tf.concat([tf.zeros([1], dtype=tf.int32), tf.cumsum(seqlens, axis=0)], axis=0), dtype=tf.int32)

# Ground truth answer: [   0,  434, 512, 613, 1024]
# Conversion: Use .numpy() to print the array values like PyTorch
print(f"cu_seqlens={cu_seqlens.numpy()}")
```