import torch
import tensorflow as tf
from tensorflow.keras import backend as K

# Leverage the similar API: tf.keras.backend.floatx
# This sets the global default float type, mimicking the dtype=torch.bfloat16 in the bug report.
K.set_floatx('bfloat16')

# Mimic torch.compile with backend="inductor" using tf.function(jit_compile=True)
@tf.function(jit_compile=True)
def compiled_gqa_attention(q, k, v):
    """
    TensorFlow implementation of Grouped Query Attention (GQA) logic.
    Preserves the logic of the original bug: Q has more heads than K/V.
    """
    # Original shapes: q=[2, 32, ...], k=[2, 8, ...]
    # We need to repeat K and V to match Q's heads for the attention calculation.
    # Assuming heads_q is a multiple of heads_kv.
    heads_q = tf.shape(q)[1]
    heads_kv = tf.shape(k)[1]
    repeats = heads_q // heads_kv
    
    # Repeat K and V along the head dimension
    k_repeated = tf.repeat(k, repeats=repeats, axis=1)
    v_repeated = tf.repeat(v, repeats=repeats, axis=1)
    
    # Scaled Dot-Product Attention
    # (Batch, Heads, Seq, HeadDim) x (Batch, Heads, HeadDim, Seq) -> (Batch, Heads, Seq, Seq)
    attn_scores = tf.matmul(q, k_repeated, transpose_b=True)
    
    # Scale factor (sqrt of head_dim)
    # Cast to float32 for sqrt to maintain precision, then cast back if necessary
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scale = tf.math.sqrt(dk)
    attn_scores = attn_scores / tf.cast(scale, q.dtype)
    
    # Softmax
    attn_weights = tf.nn.softmax(attn_scores, axis=-1)
    
    # (Batch, Heads, Seq, Seq) x (Batch, Heads, Seq, HeadDim) -> (Batch, Heads, Seq, HeadDim)
    output = tf.matmul(attn_weights, v_repeated)
    return output

def test_flex_attention_gqa_bfloat16():
    # Reproduce the tensor shapes from the bug report (scaled down for minimal test)
    # Original: [2, 32, 4096, 128] and [2, 8, 4096, 128]
    batch_size = 2
    seq_len = 128  # Reduced from 4096 for speed
    head_dim = 64  # Reduced from 128 for speed
    heads_q = 4    # Reduced from 32
    heads_kv = 2   # Reduced from 8 (GQA ratio 2:1)

    # Create tensors. Note: We don't specify dtype here to rely on K.floatx() (the similar API)
    q = tf.random.normal([batch_size, heads_q, seq_len, head_dim])
    k = tf.random.normal([batch_size, heads_kv, seq_len, head_dim])
    v = tf.random.normal([batch_size, heads_kv, seq_len, head_dim])

    # Forward pass (Compiled)
    y = compiled_gqa_attention(q, k, v)

    # Backward pass (GradientTape mimics y.backward())
    with tf.GradientTape() as tape:
        tape.watch([q, k, v])
        y = compiled_gqa_attention(q, k, v)
    
    grads = tape.gradient(y, [q, k, v])

    # Assertions to ensure execution and gradient flow
    assert y is not None
    assert grads[0] is not None
    assert grads[1] is not None
    assert grads[2] is not None
    
    # Verify dtype matches the API setting
    assert y.dtype == tf.bfloat16, f"Expected bfloat16, got {y.dtype}"

if __name__ == "__main__":
    test_flex_attention_gqa_bfloat16()