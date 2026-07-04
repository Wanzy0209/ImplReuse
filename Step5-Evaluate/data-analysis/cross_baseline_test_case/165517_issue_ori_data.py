```python
import tensorflow as tf
from einops import rearrange
import math

def qwen_image_flash_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    # TensorFlow's scaled_dot_product_attention expects [batch, seq, heads, dim]
    # Source inputs are [batch, heads, seq, dim], so we rearrange upfront.
    q = rearrange(q, "b n s d -> b s n d", n=num_heads)
    k = rearrange(k, "b n s d -> b s n d", n=num_heads)
    v = rearrange(v, "b n s d -> b s n d", n=num_heads)

    # In TensorFlow, tf.nn.scaled_dot_product_attention is the optimized equivalent
    # to Flash Attention. We map the logic to TF's native implementation.
    if enable_fp8_attention:
        # FP8 logic: manual implementation to handle scaling and casting
        origin_dtype = q.dtype
        q_std, k_std, v_std = tf.math.reduce_std(q), tf.math.reduce_std(k), tf.math.reduce_std(v)
        
        # Normalize and cast to float8
        q_fp8 = tf.cast(q / q_std, tf.float8_e4m3fn)
        k_fp8 = tf.cast(k / k_std, tf.float8_e4m3fn)
        v_fp8 = tf.cast(v / v_std, tf.float8_e4m3fn)

        # Manual Scaled Dot-Product Attention for FP8
        # QK^T
        attn_scores = tf.matmul(q_fp8, k_fp8, transpose_b=True)
        
        # Scale
        dim = tf.cast(tf.shape(q)[-1], tf.float32)
        scale = q_std * k_std / tf.math.sqrt(dim)
        attn_scores = attn_scores * scale

        # Masking
        if attention_mask is not None:
            if attention_mask.dtype == tf.bool:
                # TF boolean mask: True=keep, False=mask
                attn_scores = tf.where(attention_mask, attn_scores, tf.cast(-1e9, attn_scores.dtype))
            else:
                # Float mask: additive
                attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)

        # Matmul with V
        x = tf.matmul(attn_weights, v_fp8)

        # Cast back and rescale
        x = tf.cast(x, origin_dtype) * v_std
    else:
        # Standard path: use TF's optimized SDPA
        x = tf.nn.scaled_dot_product_attention(q, k, v, attention_mask=attention_mask)

    # Rearrange back to [batch, seq, heads * dim]
    x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    return x
```