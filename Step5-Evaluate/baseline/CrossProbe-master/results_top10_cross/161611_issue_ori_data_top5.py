```python
import tensorflow as tf

# Conversion: torch.zeros -> tf.zeros
# Note: Device placement in TensorFlow is typically handled via 'with tf.device(...):' context,
# not as an argument to the tensor constructor.
attn_bias = tf.zeros((L, S), dtype=query.dtype)  # Already query.dtype
if is_causal:
    # ...
    # Conversion: masked_fill_ -> tf.where
    # PyTorch fills where mask is True. Here mask is logical_not(temp_mask), so fill where temp_mask is False.
    attn_bias = tf.where(temp_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
    # Conversion: .to() -> tf.cast
    # Note: The original code did not assign the result, making this line a no-op.
    tf.cast(attn_bias, query.dtype)  # Redundant! Already query.dtype, and result not assigned
```