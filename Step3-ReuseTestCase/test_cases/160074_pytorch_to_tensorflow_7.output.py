import torch
import tensorflow as tf
import numpy as np

def test_tf_attention_backward_with_name_scope():
    """
    Adapted test case for TensorFlow based on the PyTorch FlexAttention bug report.
    
    Original Issue: FlexAttention backward compilation failure with GQA on NVIDIA B200.
    Original API: torch.compile (with backend="inductor")
    Target API: tf.keras.backend.name_scope
    
    This test mimics the logic of running an attention mechanism inside a 
    compiled/graph context (tf.function) wrapped by the target API (name_scope),
    followed by a gradient calculation (backward pass).
    """
    
    # Ensure GPU is available if possible, mirroring the original "cuda" context
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    # Enable mixed precision (bfloat16) to match the original dtype
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
    except ValueError:
        print("Bfloat16 not supported on this device, falling back to float32.")

    # Define the attention model
    class AttentionModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Using MultiHeadAttention as the closest equivalent to FlexAttention.
            # Note: Standard TF MHA assumes equal heads for Q, K, V. 
            # GQA (Grouped Query Attention) in TF often requires custom implementations 
            # or specific tensor manipulations, but we use the standard layer here 
            # to test the API interaction and compilation stability.
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=32, 
                key_dim=128
            )

        # tf.function acts as the compilation equivalent to torch.compile
        @tf.function(jit_compile=True) 
        def call(self, q, k, v):
            # Use the requested API: tf.keras.backend.name_scope
            # This wraps the operation similar to how a compiled graph scope works.
            with tf.keras.backend.name_scope("flex_attention_sim"):
                
                # PyTorch inputs were [Batch, Heads, Seq, HeadDim]
                # TF expects [Batch, Seq, Features]
                batch_size = tf.shape(q)[0]
                seq_len = tf.shape(q)[2]
                
                # Reshape Q: [2, 32, 4096, 128] -> [2, 4096, 4096]
                q_reshaped = tf.reshape(q, [batch_size, seq_len, -1])
                
                # Reshape K/V: [2, 8, 4096, 128] -> [2, 4096, 1024]
                k_reshaped = tf.reshape(k, [batch_size, seq_len, -1])
                v_reshaped = tf.reshape(v, [batch_size, seq_len, -1])

                # Perform attention
                attn_output = self.attention(
                    query=q_reshaped,
                    value=v_reshaped,
                    key=k_reshaped
                )
                return attn_output

    with tf.device(device):
        # Initialize model
        model = AttentionModel()

        # Create inputs matching the original PyTorch shapes
        # q: [2, 32, 4096, 128], k/v: [2, 8, 4096, 128]
        q = tf.random.normal([2, 32, 4096, 128])
        k = tf.random.normal([2, 8, 4096, 128])
        v = tf.random.normal([2, 8, 4096, 128])

        # Forward pass (triggering compilation via tf.function)
        y = model(q, k, v)

        # Backward pass (GradientTape equivalent to y.backward)
        with tf.GradientTape() as tape:
            y = model(q, k, v)
            loss = tf.reduce_sum(y)
        
        grads = tape.gradient(loss, model.trainable_variables)

        # Basic assertion to ensure execution completed
        assert grads is not None
        assert y.shape == (2, 4096, 4096) # Output shape check
        
        print("Test passed: Forward and backward execution completed within name_scope.")

if __name__ == "__main__":
    test_tf_attention_backward_with_name_scope()