```python
import tensorflow as tf
import math
from einops import rearrange

# Flash Attention 3 is not available in TensorFlow via the same interface
FLASH_ATTN_3_AVAILABLE = False

def qwen_image_flash_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_heads: int, attention_mask=None, enable_fp8_attention: bool = False):
    if FLASH_ATTN_3_AVAILABLE and attention_mask is None:
        # Conversion: TensorFlow does not have flash_attn_interface.
        # Using standard implementation.
        # The FP8 logic is omitted as it requires specific kernel support not exposed in TF.
        x = _scaled_dot_product_attention(q, k, v, attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        # Conversion: torch.nn.functional.scaled_dot_product_attention
        # implemented using standard TensorFlow ops.
        x = _scaled_dot_product_attention(q, k, v, attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x

def _scaled_dot_product_attention(q, k, v, attn_mask=None):
    # Transpose k from (B, N, S, D) to (B, N, D, S)
    kt = tf.transpose(k, perm=[0, 1, 3, 2])
    # QK^T
    qk = tf.matmul(q, kt)
    # Scale
    scale = tf.math.sqrt(tf.cast(tf.shape(q)[-1], qk.dtype))
    logits = qk / scale

    # Mask handling
    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            # PyTorch: True means masked. TF: add -inf.
            # Note: PyTorch SDPA expects boolean mask where True indicates elements to mask.
            logits = tf.where(attn_mask, -1e9, logits)
        else:
            # Additive mask
            logits += attn_mask

    # Softmax
    weights = tf.nn.softmax(logits, axis=-1)
    # Matmul with V
    output = tf.matmul(weights, v)
    return output
```