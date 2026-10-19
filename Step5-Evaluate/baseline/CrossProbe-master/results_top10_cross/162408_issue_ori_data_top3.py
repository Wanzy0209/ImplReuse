```python
import tensorflow as tf

# Conversion: TF does not have a global default dtype setting that affects tensor constructors in the same way.
# torch.set_default_dtype(torch.bfloat16)

device = "/CPU:0"
# device="/GPU:0"

# Conversion: Use tf.device context to handle device placement
with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

    # Conversion: torch.zeros -> tf.zeros, torch.cumsum -> tf.cumsum, torch.cat -> tf.concat
    # Note: We explicitly use int32 for zeros to match the final .to(dtype=torch.int32) in the source.
    # The original source code would likely raise a dtype mismatch error in PyTorch without casting the zeros first.
    cu_seqlens = tf.concat([tf.zeros([1], dtype=tf.int32), tf.cumsum(seqlens, axis=0)], axis=0)

# Ground truth answer: [   0,  434,  512,  613, 1024]
# Conversion: Use .numpy() to print the actual values in eager execution
print(f"{cu_seqlens.numpy()=}")
```