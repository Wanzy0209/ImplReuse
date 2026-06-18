import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class CausalAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(CausalAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        
        # Map PyTorch nn.Linear to tf.keras.layers.Dense
        # use_bias=False matches the PyTorch definition
        self.values = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.keys = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.queries = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.fc_out = tf.keras.layers.Dense(embed_size)

    def call(self, values, keys, query, mask):
        N = tf.shape(query)[0]
        value_len = tf.shape(values)[1]
        key_len = tf.shape(keys)[1]
        query_len = tf.shape(query)[1]

        # Reshape inputs to split heads
        # PyTorch: values.reshape(N, value_len, self.heads, self.head_dim)
        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

        # Pass through linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum for energy calculation: "nqhd,nkhd->nhqk"
        energy = tf.einsum("nqhd,nkhd->nhqk", queries, keys)

        # Apply mask
        # PyTorch: energy.masked_fill(mask == 0, float("-1e20"))
        # TensorFlow equivalent using tf.where
        energy = tf.where(mask == 0, -1e20, energy)

        # Softmax attention
        # PyTorch: torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # TensorFlow: tf.nn.softmax(..., axis=-1) (last axis is dim 3 here)
        attention = tf.nn.softmax(energy / tf.sqrt(tf.cast(self.embed_size, tf.float32)), axis=-1)

        # Einsum for output: "nhql,nlhd->nqhd"
        out = tf.einsum("nhql,nlhd->nqhd", attention, values)
        
        # Reshape and final linear layer
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))
        out = self.fc_out(out)
        return out

def test_tpu_rewrite_numerical_consistency():
    # Model parameters
    embed_size = 256
    heads = 8
    batch_size = 2
    seq_len = 10
    
    # Initialize the layer
    attention_layer = CausalAttention(embed_size, heads)
    
    # Create dummy inputs
    # Using random normal to simulate data
    values = tf.random.normal((batch_size, seq_len, embed_size))
    keys = tf.random.normal((batch_size, seq_len, embed_size))
    query = tf.random.normal((batch_size, seq_len, embed_size))
    
    # Create a causal mask (upper triangular)
    # 1 for valid, 0 for invalid
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
    mask = tf.tile(mask, [batch_size, heads, 1, 1])

    # 1. Run in Eager Mode (Baseline)
    eager_output = attention_layer(values, keys, query, mask)

    # 2. Run using tf.compat.v1.tpu.rewrite
    # Define the computation function for the rewrite
    def computation_fn(v, k, q, m):
        return attention_layer(v, k, q, m)

    # Note: tf.compat.v1.tpu.rewrite requires a TPU environment.
    # We wrap the execution in a try-except block to handle environments without TPUs gracefully,
    # but the core logic tests the API as requested.
    try:
        # Initialize TPU system (Required for tf.compat.v1.tpu.rewrite)
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        print("TPU initialized. Running tf.compat.v1.tpu.rewrite...")
        
        # Execute the rewritten computation
        # The API returns a list of tensors corresponding to the function's return values
        rewritten_outputs = tf.compat.v1.tpu.rewrite(
            computation_fn, 
            inputs=[values, keys, query, mask]
        )
        rewritten_output = rewritten_outputs[0]

        # 3. Verify Numerical Consistency
        # The bug report highlights "severe numerical inconsistencies".
        # We check if the outputs are close within a reasonable tolerance.
        
        # Convert to numpy for comparison
        eager_np = eager_output.numpy()
        rewritten_np = rewritten_output.numpy()

        # Assert closeness
        np.testing.assert_allclose(
            eager_np, 
            rewritten_np, 
            rtol=1e-5, 
            atol=1e-5,
            err_msg="Numerical inconsistency detected between Eager and TPU Rewrite execution."
        )
        print("Test Passed: Numerical consistency verified.")

    except tf.errors.NotFoundError:
        print("Skipping TPU rewrite test: No TPU hardware found.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_tpu_rewrite_numerical_consistency()