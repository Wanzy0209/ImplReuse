```python
import tensorflow as tf

# TensorFlow does not support setting a global default dtype in the same way as PyTorch.
# Dtypes must be specified explicitly during tensor creation.
# torch.set_default_dtype(torch.bfloat16)

device = "cpu"
# device="cuda" # In TensorFlow, device placement is handled implicitly or via tf.device contexts

# tf.constant replaces torch.tensor. Note that device is not passed in the constructor.
seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

# Conversion logic:
# 1. torch.zeros -> tf.zeros (explicitly set dtype=tf.int32 to match the final cast and avoid concat errors)
# 2. torch.cumsum(..., dim=0) -> tf.cumsum(..., axis=0)
# 3. torch.cat(...) -> tf.concat(..., axis=0) (takes a list instead of tuple)
# 4. .to(dtype=...) -> tf.cast(..., dtype=...)
cu_seqlens = tf.cast(
    tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(seqlens, axis=0)], axis=0),
    dtype=tf.int32
)

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"cu_seqlens={cu_seqlens}")
```