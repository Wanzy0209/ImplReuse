```python
import tensorflow as tf

def qwen_image_flash_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_heads: int, attention_mask=None, enable_fp8_attention: bool = False):
    # Helper function to mimic torch.nn.functional.scaled_dot_product_attention
    def scaled_dot_product_attention(q, k, v, attn_mask=None):
        # q, k, v shape: (batch, heads, seq_len, depth)
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], q.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if attn_mask is not None:
            scaled_attention_logits += attn_mask

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    # Note: FLASH_ATTN_3_AVAILABLE is PyTorch specific. 
    # In TensorFlow, we rely on XLA for efficient attention kernels.
    # We preserve the structure: if mask is None, we take the optimized path.
    if attention_mask is None:
        if not enable_fp8_attention:
            # Conversion: rearr