```python
import tensorflow as tf

# torch.set_default_dtype(torch.bfloat16) # TF does not support global default dtype in this manner
device = "cpu"
# device="cuda"
seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)
# Conversion note: Original PyTorch code had a type mismatch (float zeros vs int cumsum).
# Fixed by explicitly casting zeros to int32 to match the final output type.
cu_seqlens = tf.concat([tf.zeros([1], dtype=tf.int32), tf.cumsum(seqlens, axis=0)], axis=0)

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"{cu_seqlens=}")
```