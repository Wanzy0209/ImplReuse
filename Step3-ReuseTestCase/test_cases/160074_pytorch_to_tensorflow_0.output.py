import torch
import tensorflow as tf
import sys

def test_flex_attention_gqa_backward():
    """
    Test case adapted from PyTorch FlexAttention bug report (Issue 160074).
    Verifies the behavior of tf.compat.v1.tpu.rewrite with GQA (Grouped Query Attention)
    logic and backward pass (gradient computation).
    """
    
    # Initialize TPU system
    # Note: This requires a TPU environment. If not available, this will raise an error,
    # which is expected behavior for TPU-specific APIs.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU system initialized.")
    except Exception as e:
        print(f"Skipping TPU initialization (expected if no TPU hardware): {e}")
        # We proceed to define the test, but execution will fail on hardware without TPU.
        # This is consistent with testing hardware-specific compilation APIs.

    # Define the computation function mimicking FlexAttention with GQA
    def flex_attention_computation(q, k, v):
        """
        Simulates FlexAttention with GQA enabled.
        q: [Batch, Heads_Q, Seq, Dim]
        k: [Batch, Heads_KV, Seq, Dim]
        v: [Batch, Heads_KV, Seq, Dim]
        """
        # GQA Logic: Heads_Q (32) > Heads_KV (8).
        # We repeat K and V to match Q heads to simulate the grouped query behavior.
        # In the bug report: 32 / 8 = 4.
        k_repeated = tf.repeat(k, repeats=4, axis=1)
        v_repeated = tf.repeat(v, repeats=4, axis=1)

        # Scaled Dot-Product Attention
        # Transpose k: [Batch, Heads_Q, Dim, Seq]
        kt = tf.transpose(k_repeated, [0, 1, 3, 2])

        # Matmul: [Batch, Heads_Q, Seq, Seq]
        scores = tf.matmul(q, kt)

        # Scale
        scale = tf.math.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
        scores = scores / scale

        # Softmax
        attn_weights = tf.nn.softmax(scores, axis=-1)

        # Matmul with v: [Batch, Heads_Q, Seq, Dim]
        output = tf.matmul(attn_weights, v_repeated)

        return output

    # Define shapes matching the bug report
    # Batch=2, Seq=4096, Heads_Q=32, Heads_KV=8, Dim=128
    batch_size = 2
    seq_len = 4096
    heads_q = 32
    heads_kv = 8
    head_dim = 128

    # Create inputs with bfloat16 (matching the bug report)
    # requires_grad=True is implicit in TF when watching tensors in GradientTape
    q = tf.random.normal([batch_size, heads_q, seq_len, head_dim], dtype=tf.bfloat16)
    k = tf.random.normal([batch_size, heads_kv, seq_len, head_dim], dtype=tf.bfloat16)
    v = tf.random.normal([batch_size, heads_kv, seq_len, head_dim], dtype=tf.bfloat16)

    # Use GradientTape to track operations for the backward pass
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q)
        tape.watch(k)
        tape.watch(v)

        # Call tf.compat.v1.tpu.rewrite
        # This compiles the computation for TPU (similar to torch.compile with backend="inductor")
        y = tf.compat.v1.tpu.rewrite(
            flex_attention_computation,
            inputs=[q, k, v]
        )

    # Perform backward pass
    # Mimics y.backward(torch.randn_like(y))
    grad_y = tf.random.normal(tf.shape(y), dtype=tf.bfloat16)
    
    try:
        grads = tape.gradient(y, [q, k, v], output_gradients=grad_y)
        
        # Assertions to verify gradients were computed
        assert grads[0] is not None, "Gradient for q is None"
        assert grads[1] is not None, "Gradient for k is None"
        assert grads[2] is not None, "Gradient for v is None"
        
        print("Test Passed: Forward and Backward execution succeeded on TPU.")
        
    except Exception as e:
        print(f"Test Failed: Error during backward pass or compilation: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_gqa_backward()