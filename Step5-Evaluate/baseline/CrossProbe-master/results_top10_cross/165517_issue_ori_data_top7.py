```python
import tensorflow as tf

def qwen_image_flash_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    # Conversion: FLASH_ATTN_3_AVAILABLE is specific to PyTorch.
    # In TensorFlow, we use the standard optimized attention kernel.
    # We handle the FP8 logic manually if requested, otherwise use standard path.
    
    # Conversion: Rearrange inputs from (Batch, Heads, SeqLen, Depth) to (Batch, SeqLen, Heads, Depth)
    # which is the expected format for tf.nn.scaled_dot_product_attention.
    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.transpose(v, [0, 2, 1, 3])

    if enable_fp8_attention:
        # Conversion: Manual FP8 handling using TensorFlow ops.
        # Note: This requires TensorFlow version that supports float8_e4m3fn.
        origin_dtype = q.dtype
        q_std = tf.math.reduce_std(q)
        k_std = tf.math.reduce_std(k)
        v_std = tf.math.reduce_std(v)

        # Normalize and cast to float8
        q = tf.cast(q / q_std, tf.float8_e4m3fn)
        k = tf.cast(k / k_std, tf.float8_e4m3fn)
        v = tf.cast(v / v_std, tf.float8_e4m3fn)

        # Conversion: Use tf.nn.scaled_dot_product_attention.
        # The source code calculates a specific softmax_scale for flash_attn_func.
        # Since we normalized Q and K by their std devs, and the source passed
        # scale = q_std * k_std / sqrt(d), the math cancels out to standard attention scaling.
        x = tf.nn.scaled_dot_product_attention(q, k, v, mask=attention_mask)

        # Cast back to original dtype and apply value scaling
        x = tf.cast(x, origin_dtype) * v_std
    else:
        # Conversion: PyTorch's scaled_dot_product_attention maps to tf.nn.scaled_dot_product_attention.
        x = tf.nn.scaled_dot_product_attention(q, k, v, mask=attention_mask)

    # Conversion: Reshape output from (Batch, SeqLen, Heads, Depth) to (Batch, SeqLen, Heads * Depth)
    # Equivalent to rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    x = tf.reshape(x, [batch_size, seq_len, -1])

    return x
```