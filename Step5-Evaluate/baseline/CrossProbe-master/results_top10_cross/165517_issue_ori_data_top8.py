```python
import tensorflow as tf
from einops import rearrange
import math

# Conversion: Flash Attention is a PyTorch-specific library.
# We define this as False to use the standard implementation path.
FLASH_ATTN_3_AVAILABLE = False

def qwen_image_flash_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    if FLASH_ATTN_3_AVAILABLE and attention_mask is None:
        if not enable_fp8_attention:
            # Conversion: Rearrange tensors to match FlashAttn expected input (b, s, n, d)
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            
            # Conversion: Replacing flash_attn_func with standard TF attention ops
            # Transpose K for matmul: (b, s, n, d) -> (b, s, d, n)
            kt = tf.transpose(k, perm=[0, 1, 3, 2])
            # Matmul: (b, s, n, d) * (b, s, d, n) -> (b, s, n, n)
            qk = tf.matmul(q, kt)
            # Scale
            scale = tf.math.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
            qk = qk / scale
            # Softmax
            attn_weights = tf.nn.softmax(qk, axis=-1)
            # Matmul with V: (b, s, n, n) * (b, s, n, d) -> (b, s, n, d)
            x = tf.matmul(attn_weights, v)
            
            if isinstance(x, tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            origin_dtype = q.dtype
            # Conversion: Calculate standard deviation
            q_std, k_std, v_std = tf.math.reduce_std(q), tf.math.reduce_std(k), tf.math.reduce_std(v)
            
            # Conversion: Cast to float8 (TF supports tf.float8_e4m3fn)
            q = tf.cast(q / q_std, tf.float8_e4m3fn)
            k = tf.cast(k / k_std, tf.float8_e4m3fn)
            v = tf.cast(v / v_std, tf.float8_e4m3fn)
            
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            
            # Conversion: Replacing flash_attn_func. 
            # Note: Standard TF matmul might not support float8 on all devices, 
            # so we cast back to origin_dtype for computation to ensure validity.
            q_f = tf.cast(q, origin_dtype)
            k_f = tf.cast(k, origin_dtype)
            v_f = tf.cast(v, origin_dtype)
            
            kt = tf.transpose(k_f, perm=[0, 1, 3, 2])
            qk = tf.matmul(q_f, kt)
            
            # Apply softmax scale
            scale_factor = q_std * k_std / tf.math.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
            qk = qk * scale_factor
            
            attn_weights = tf.nn.softmax(qk, axis=-1)
            x = tf.matmul(attn_weights, v_f)
            
            # Cast back and scale
            x = tf.cast(x, origin_dtype) * v_std
            if isinstance(x, tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        # Conversion: torch.nn.functional.scaled_dot_product_attention
        # Inputs are (b, n, s, d)
        
        # Transpose K to (b, n, d, s)
        kt = tf.transpose(k, perm=[0, 1, 3, 2])
        
        # Matmul QK: (b, n, s, d) @ (b, n, d, s) -> (b, n, s, s)
        qk = tf.matmul(q, kt)
        
        # Scale
        d_k = tf.cast(tf.shape(q)[-1], qk.dtype)
        qk = qk / tf.math.sqrt(d_k)
        
        # Conversion: Handle attention_mask
        if attention_mask is not None:
            # If mask is boolean, convert to additive (-inf for False)
            if attention_mask.dtype == tf.bool:
                qk = tf.where(attention_mask, qk, tf.fill(tf.shape(qk), -1e9))
            else:
                # Assume additive mask
                qk = qk + attention_mask
        
        # Softmax
        attn_weights = tf.nn.softmax(qk, axis=-1)
        
        # Matmul with V: (b, n, s, s) @ (b, n, s, d) -> (b, n, s, d)
        x = tf.matmul(attn_weights, v)
        
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x
```