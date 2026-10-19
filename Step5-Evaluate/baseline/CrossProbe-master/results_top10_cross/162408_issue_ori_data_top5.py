```python
import tensorflow as tf

# torch.set_default_dtype(torch.bfloat16)
# Note: TensorFlow does not have a global default dtype setting that affects tensor creation
# in the same way as PyTorch. Dtypes are typically specified explicitly during tensor creation.
device = "/cpu:0"
# device="/gpu:0"

with tf.device(device):
    # torch.tensor(..., device=device, dtype=torch.int32)
    seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

    # torch.cat((torch.zeros(1, device=device), torch.cumsum(seqlens, dim=0))).to(...)
    # Conversion: tf.concat takes a list of tensors, tf.cumsum uses 'axis' instead of 'dim',
    # and tf.cast is used for dtype conversion.
    cu_seqlens = tf.cast(
        tf.concat([tf.zeros([1]), tf.cumsum(seqlens, axis=0)], axis=0),
        dtype=tf.int32
    )

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"{cu_seqlens=}")
```