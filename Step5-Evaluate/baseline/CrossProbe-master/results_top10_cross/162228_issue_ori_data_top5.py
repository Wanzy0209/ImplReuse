```python
import tensorflow as tf

# Define a helper function to mimic PyTorch's flex_attention behavior
# Note: TensorFlow does not have a direct 1:1 equivalent for flex_attention.
# We implement a standard Scaled Dot-Product Attention that applies the score_mod.
def flex_attention(q, k, v, score_mod):
    # q, k, v shape: (B, L, H, D)
    B, L, H, D = tf.unstack(tf.shape(q))

    # S