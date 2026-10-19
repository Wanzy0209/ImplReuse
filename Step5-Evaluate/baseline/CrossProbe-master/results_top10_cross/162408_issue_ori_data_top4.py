```python
import tensorflow as tf

# torch.set_default_dtype(torch.bfloat16)
# Conversion: TensorFlow does not support setting a global default dtype in this manner.
# Dtypes are typically specified explicitly during tensor creation.
device = "cpu"
# device="cuda"
# Conversion: In TensorFlow, device placement is handled by the runtime or tf.device contexts.
seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

# Conversion: torch.cat -> tf.concat, torch.zeros -> tf.zeros, torch.cumsum -> tf.cumsum
# Note: Explicitly setting dtype=tf.int32 for zeros to match the .to(dtype=torch.int32) logic
cu_seqlens = tf.concat([tf.zeros([1], dtype=tf.int32), tf.cumsum(seqlens, axis=0)], axis=0)

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"cu_seqlens={cu_seqlens}")
```