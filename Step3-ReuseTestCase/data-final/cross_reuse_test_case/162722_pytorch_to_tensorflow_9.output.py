import torch
import tensorflow as tf
import numpy as np

class CausalAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(CausalAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        
        # Using name_scope to organize variable creation
        with tf.name_scope("attention_weights"):
            self.values = tf.keras.layers.Dense(self.head_dim, use_bias=False)
            self.keys = tf.keras.layers.Dense(self.head_dim, use_bias=False)
            self.queries = tf.keras.layers.Dense(self.head_dim, use_bias=False)
            self.fc_out = tf.keras.layers.Dense(embed_size)

    def call(self, values, keys, query, mask):
        # Using name_scope to organize the forward pass operations in the graph
        with tf.name_scope("causal_attention_forward"):
            N = tf.shape(query)[0]
            value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(query)[1]

            # Reshape for multi-head attention
            with tf.name_scope("reshape_heads"):
                values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
                keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
                queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

            # Einstein summation for energy calculation
            with tf.name_scope("energy_calculation"):
                energy = tf.einsum("nqhd,nkhd->nhqk", queries, keys)

            # Masking
            if mask is not None:
                with tf.name_scope("masking"):
                    # TensorFlow equivalent of masked_fill
                    energy = tf.where(mask == 0, -1e20, energy)

            with tf.name_scope("softmax"):
                attention = tf.nn.softmax(energy / tf.math.sqrt(tf.cast(self.embed_size, tf.float32)), axis=3)

            # Output calculation
            with tf.name_scope("output_projection"):
                out = tf.einsum("nhql,nlhd->nqhd", attention, values)
                out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))
                out = self.fc_out(out)

            return out

class CausalAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(CausalAttentionBlock, self).__init__()
        with tf.name_scope("transformer_block_init"):
            self.attention = CausalAttention(embed_size, heads)
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            
            self.feed_forward = tf.keras.Sequential([
                tf.keras.layers.Dense(embed_size * forward_expansion, activation='relu'),
                tf.keras.layers.Dense(embed_size)
            ])
            self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, values, keys, query, mask):
        with tf.name_scope("transformer_block_forward"):
            # Attention and Residual Connection
            with tf.name_scope("attention_residual"):
                attn = self.attention(values, keys, query, mask)
                x = self.dropout(attn)
                x = self.norm1(x + query)
            
            # Feed Forward and Residual Connection
            with tf.name_scope("ffn_residual"):
                forward = self.feed_forward(x)
                out = self.dropout(forward)
                out = self.norm2(out + x)
            
            return out

class CausalAttentionDNN(tf.keras.Model):
    def __init__(self, input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length):
        super(CausalAttentionDNN, self).__init__()
        with tf.name_scope("model_init"):
            self.embed_size = embed_size
            self.word_embedding = tf.keras.layers.Embedding(input_size, embed_size)
            self.position_embedding = tf.keras.layers.Embedding(max_length, embed_size)
            
            self.layers = [
                CausalAttentionBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
            
            self.fc = tf.keras.layers.Dense(output_size)
            self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask):
        with tf.name_scope("model_forward"):
            N, seq_length = tf.shape(x)[0], tf.shape(x)[1]
            
            # Positional encoding
            positions = tf.range(start=0, limit=seq_length, delta=1)
            positions = tf.tile(tf.expand_dims(positions, 0), [N, 1])
            
            with tf.name_scope("embedding_add"):
                out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
            
            # Pass through transformer blocks
            for layer in self.layers:
                out = layer(out, out, out, mask)
            
            out = self.fc(out)
            return out

# Test Case
def test_tf_name_scope_transformer():
    # Define hyperparameters
    input_size = 100
    embed_size = 256
    num_layers = 2
    heads = 8
    forward_expansion = 4
    output_size = 10
    dropout = 0.1
    max_length = 50
    batch_size = 4
    seq_len = 10

    # Initialize model
    model = CausalAttentionDNN(
        input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length
    )

    # Create dummy input and mask
    x = tf.random.uniform((batch_size, seq_len), minval=0, maxval=input_size, dtype=tf.int32)
    mask = tf.ones((batch_size, 1, 1, seq_len)) # Simple mask allowing all positions

    # Run forward pass
    # Note: tf.name_scope is primarily for graph organization and visualization (TensorBoard).
    # Unlike torch.compile, it does not alter the numerical execution path or optimization.
    # This test verifies that the complex Transformer logic executes correctly within the scopes.
    output = model(x, mask)

    # Assertions
    assert output.shape == (batch_size, seq_len, output_size), f"Output shape mismatch: {output.shape}"
    assert not tf.reduce_any(tf.math.is_nan(output)), "Output contains NaN values"
    
    print("Test passed: Model executed successfully with tf.name_scope structure.")

if __name__ == "__main__":
    test_tf_name_scope_transformer()