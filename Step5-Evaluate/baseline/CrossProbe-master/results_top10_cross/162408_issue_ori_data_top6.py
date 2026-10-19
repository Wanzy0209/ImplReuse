```python
import tensorflow as tf

# Conversion Note: PyTorch's set_default_dtype is a global setting.
# TensorFlow typically handles dtypes explicitly per operation or via a policy context.
# Since the following operations are on int32 tensors, this setting does not affect the logic.
# torch.set_default_dtype(torch.bfloat16)

device = "/cpu:0"
# device = "/gpu:0"

with tf.device(device):
    # torch.tensor([434,  78, 101, 411], device=device, dtype=torch.int32)
    seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

    # torch.zeros(1, device=device)
    zeros = tf.zeros([1], dtype=tf.int32)

    # torch.cumsum(seqlens, dim=0)
    cumsum = tf.cumsum(seqlens, axis=0)

    # torch.cat((zeros, cumsum))
    cu_seqlens = tf.concat([zeros, cumsum], axis=0)

    # .to(dtype=torch.int32, device=device)
    # Explicit cast to ensure int32, though inputs were already int32
    cu_seqlens = tf.cast(cu_seqlens, dtype=tf.int32)

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"{cu_seqlens=}")
```