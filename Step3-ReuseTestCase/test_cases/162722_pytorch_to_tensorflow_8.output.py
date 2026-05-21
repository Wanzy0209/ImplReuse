import torch
import tensorflow as tf
import numpy as np

class CausalAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, name="causal_attention"):
        super(CausalAttention, self).__init__(name=name)
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Using tf.keras.name_scope to organize variable creation, 
        # mirroring the structure of the original PyTorch model.
        with tf.keras.name_scope("projections"):
            self.values = tf.keras.layers.Dense(self.head_dim, use_bias=False)
            self.keys = tf.keras.layers.Dense(self.head_dim, use_bias=False)
            self.queries = tf.keras.layers.Dense(self.head_dim, use_bias=False)
            self.fc_out = tf.keras.layers.Dense(embed_size)

    def call(self, values, keys, query, mask):
        N = tf.shape(query)[0]
        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(query)[1]

        # Reshape for multi-head attention
        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum for energy calculation
        energy = tf.einsum("nqhd,nkhd->nhqk", queries, keys)

        # Masking
        if mask is not None:
            # Apply mask by setting energy to a very low number where mask is 0
            energy = tf.where(mask == 0, -1e20, energy)

        # Attention
        attention = tf.nn.softmax(energy / tf.math.sqrt(tf.cast(self.embed_size, tf.float32)), axis=3)

        # Output calculation
        out = tf.einsum("nhql,nlhd->nqhd", attention, values)
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))

        out = self.fc_out(out)
        return out

def test_numerical_consistency():
    """
    Test case adapted to verify behavior within tf.keras.name_scope.
    The original bug report highlighted severe numerical inconsistencies 
    with torch.compile. This test verifies that the TensorFlow equivalent
    maintains numerical consistency across multiple runs.
    """
    # Model Parameters
    embed_size = 256
    heads = 8
    batch_size = 4
    seq_length = 10
    
    # Create dummy inputs
    # Using a fixed seed to ensure reproducibility of the test itself
    tf.random.set_seed(42)
    np.random.seed(42)
    
    x = tf.random.normal((batch_size, seq_length, embed_size))
    # Create a simple causal mask (lower triangular)
    mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    mask = tf.expand_dims(mask, 0) # Add batch dimension
    mask = tf.tile(mask, [batch_size, 1, 1])

    # Instantiate model inside the specific API scope
    with tf.keras.name_scope("transformer_consistency_test"):
        model = CausalAttention(embed_size, heads)
        
        # Run inference twice to check for consistency
        out1 = model(x, x, x, mask)
        out2 = model(x, x, x, mask)

    # Verify consistency
    # The original bug showed differences far beyond expected variations.
    # Here we assert that outputs are identical for the same input.
    diff = tf.reduce_max(tf.abs(out1 - out2))
    
    # Assert that the difference is negligible (numerical consistency)
    assert diff < 1e-6, f"Numerical inconsistency detected: {diff}"
    
    # Verify output shape
    assert out1.shape == (batch_size, seq_length, embed_size), \
        f"Output shape mismatch. Expected {(batch_size, seq_length, embed_size)}, got {out1.shape}"
    
    # Verify no NaNs (common symptom of numerical instability)
    assert not tf.reduce_any(tf.math.is_nan(out1)), "Output contains NaN values"
    
    print("Test passed: Model is numerically consistent within tf.keras.name_scope.")

if __name__ == "__main__":
    test_numerical_consistency()