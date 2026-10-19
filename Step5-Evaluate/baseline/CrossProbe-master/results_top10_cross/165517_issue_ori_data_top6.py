```python
import tensorflow as tf
import math
from einops import rearrange

# Conversion: Implementing torch.nn.functional.scaled_dot_product_attention equivalent
def scaled_dot_product_attention(q, k, v, attn_mask=None):
    # q, k, v shape: [batch, num_heads, seq_len, head_dim]
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    if attn_mask is not None:
        scaled_attention_logits += attn_mask

    # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, head_dim)

    return output

def qwen_image_flash_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    # Conversion: FLASH_ATTN_3_AVAILABLE is PyTorch specific. 
    # We use the standard TensorFlow implementation which is optimized via XLA/graph compilation.
    # The logic below preserves the structure but replaces the PyTorch-specific calls.
    
    if attention_mask is None:
        if not enable_fp8_attention:
            # Conversion: Rearrange logic preserved
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            
            # Conversion: flash_attn_interface.flash_attn_func replaced with TF equivalent
            # Note: Flash Attention 3 is not directly available in TF public API.
            # We use standard SDPA. For [batch, seq, heads, dim], we need to adjust the helper
            # or transpose back. The helper expects [batch, heads, seq, dim].
            # Let's transpose back for the helper to keep it simple, or adapt helper.
            # To preserve structure, let's adapt the helper call.
            # Actually, standard TF SDPA is easier with [batch, heads, seq, dim].
            # Let's transpose back to call the helper, then transpose result.
            
            q_bnsd = rearrange(q, "b s n d -> b n s d", n=num_heads)
            k_bnsd = rearrange(k, "b s n d -> b n s d", n=num_heads)
            v_bnsd = rearrange(v, "b s n d -> b n s d", n=num_heads)
            
            x = scaled_dot_product_attention(q_bnsd, k_bnsd, v_bnsd)
            
            # The original flash_attn returns [batch, seq, heads, dim] or tuple
            # We need