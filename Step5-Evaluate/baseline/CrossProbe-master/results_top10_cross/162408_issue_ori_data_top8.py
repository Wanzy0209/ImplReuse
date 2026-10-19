```python
import tensorflow as tf

# torch.set_default_dtype(torch.bfloat16) # TF does not have a global default dtype setting; dtypes are explicit
device = "cpu"
# device="cuda" # TF handles device placement automatically (e.g., via tf.device context)
seqlens = tf.constant([434,  78, 101, 411], dtype=tf.int32)

# PyTorch: torch.cat((torch.zeros(1, device=device), torch.cumsum(seqlens, dim=0))).to(...)
# Note: Explicitly creating zeros as int32 to match cumsum dtype for concatenation (original PyTorch code would raise a type error with bfloat16 default)
cu_seqlens = tf.cast(
    tf.concat(
        [tf.zeros(1, dtype=tf.int32), tf.cumsum(seqlens, axis=0)],
        axis=0
    ),
    dtype=tf.int32
)

# Ground truth answer: [   0,  434,  512,  613, 1024]
print(f"{cu_seqlens=}")
```