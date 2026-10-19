```python
import tensorflow as tf

# PyTorch: torch.zeros(L, S, dtype=query.dtype, device=query.device)
# TensorFlow: tf.zeros creates a tensor of zeros. Device placement is implicit or handled by context.
attn_bias = tf.zeros((L, S), dtype=query.dtype)

if is_causal:
    # ...
    # PyTorch: attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    # TensorFlow: Tensors are immutable. Use tf.where to create a new tensor based on the condition.
    # Fills with -inf where temp_mask is False (equivalent to logical_not(temp_mask) in PyTorch).
    attn_bias = tf.where(temp_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))

    # PyTorch: attn_bias.to(query.dtype) # Redundant! Already query.dtype, and result not assigned
    # TensorFlow: tf.cast returns a new tensor. Not assigning it makes it a no-op, matching the source behavior.
    tf.cast(attn_bias, query.dtype)  # Redundant! Already query.dtype, and result not assigned
```