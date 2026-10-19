```python
import tensorflow as tf

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # L, S = query.size(-2), key.size(-2)
    # Conversion: Use tf.shape to get dynamic tensor dimensions
    L = tf.shape(query)[-2]
    S = tf.shape(key)[-2]
    
    # scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # Conversion: Use tf.sqrt and tf.cast for tensor operations
    if scale is None:
        scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
    else:
        scale_factor = scale

    # attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    # Conversion: Use tf.zeros, device placement is handled automatically by TF
    attn_bias = tf.zeros((L, S), dtype=query.dtype)

    if is_causal:
        # assert attn_mask is None
        # temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        # attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        # attn_bias.to(query.dtype)
        
        # Conversion: Create lower triangular mask using tf.linalg.band_part
        # band_part(input, -1, 0) keeps the lower triangle
        lower_tri_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Invert mask to identify positions to mask (upper triangle)
        invalid_mask = tf.logical_not(lower_tri_mask)
        
        # Fill invalid positions with -inf
        neg_inf = tf.constant(-float('inf'), dtype=query.dtype)
        attn_bias = tf.where(invalid_mask, neg_inf, attn_bias)

    if attn_mask is not None:
        # if attn_mask.dtype == torch.bool:
        if attn_mask.dtype == tf.bool:
            # attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            # Conversion: Use tf.where to apply mask
            neg_inf = tf.constant(-float('inf'), dtype=query.dtype)
            attn_bias = tf.where(tf.logical_not(attn_mask), neg_inf, attn_bias)
        else:
            # attn_bias = attn_mask + attn_bias
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        # value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
        # Conversion: Use tf.repeat to repeat elements along the axis
        n_heads_q = tf.shape(query)[-3]
        n_heads_k = tf.shape(key)[-3]
        repeats = n_heads_q // n_heads_k
        
        key = tf.repeat(key, repeats, axis=-3)
        value = tf.repeat(value, repeats, axis=-3)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # Conversion: Use tf.matmul and tf.transpose
    # Construct permutation to swap last two dimensions
    rank = len(query.shape)
    perm = list(range(rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    
    attn_weight = tf.matmul(query, tf.transpose(key, perm)) * scale_factor
    
    # attn_weight += attn_bias
    attn_weight = attn_weight + attn_bias
    
    # attn_weight = torch.softmax(attn_weight, dim=-1)
    # Conversion: Use tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # Conversion: Use tf.nn.dropout. Note: TF dropout rate is probability of dropping, same as PyTorch p.
    if dropout_p > 0.0:
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
        
    # return attn_weight @ value
    return tf.matmul(attn_weight, value)
```