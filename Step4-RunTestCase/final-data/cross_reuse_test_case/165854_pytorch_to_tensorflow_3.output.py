import tensorflow as tf
import numpy as np

# Enable eager execution as requested by the similar API
tf.compat.v1.enable_eager_execution()

def run_with_head_count(H, device, dtype):
    """Run attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Create captured buffer that depends on dynamic H
    # In TensorFlow eager, tensors are concrete values
    head_scale = tf.random.normal([H], dtype=dtype)

    def score_mod(score, batch, head, token_q, token_kv):
        # Mimic the logic: score * head_scale[head]
        # score shape is typically (B, H, S, S)
        # head_scale shape is (H,)
        # We broadcast head_scale to multiply across the sequence dimensions
        return score * tf.reshape(head_scale, [1, H, 1, 1])

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        # Create inputs with dynamic head dimension H
        q = tf.random.normal([B, H, S, D], dtype=dtype)
        k = tf.random.normal([B, H, S, D], dtype=dtype)
        v = tf.random.normal([B, H, S, D], dtype=dtype)

        # Use GradientTape for computing gradients in eager execution
        with tf.GradientTape() as tape:
            # Watch the input tensors as they are not trainable variables
            tape.watch([q, k, v, head_scale])

            # Simplified attention mechanism to mimic flex_attention behavior
            # 1. Compute scores (QK^T)
            scores = tf.matmul(q, k, transpose_b=True)
            
            # 2. Apply score_mod (which captures the dynamic head_scale buffer)
            scores = score_mod(scores, None, None, None, None)
            
            # 3. Softmax and Output
            attn_weights = tf.nn.softmax(scores, axis=-1)
            outputs = tf.matmul(attn_weights, v)

            # Compute loss
            loss = tf.reduce_sum(outputs)
        
        # Compute gradients using GradientTape
        grads = tape.gradient(loss, [q, k, v, head_scale])

        # Verify gradients are computed (sanity check for eager execution)
        assert grads[0] is not None

    print(f"   Completed {i+1} iterations")


def main():
    # Note: TensorFlow eager execution handles device placement implicitly or via tf.device
    device = "cuda" 
    dtype = tf.float16
    # Fix: Use tf.set_seed for compatibility with TensorFlow 1.x API
    tf.set_seed(0)

    # Test with different head counts - this makes H a dynamic dimension
    # and the captured buffer (head_scale) changes size with H
    head_counts = [4, 8, 4, 16, 4]

    print(f"Running attention with dynamic head counts on {device}, dtype={dtype}")
    print(f"Testing head counts: {head_counts}\n")

    for iteration, H in enumerate(head_counts, start=1):
        print(f"Iteration {iteration}:")
        run_with_head_count(H, device, dtype)


if __name__ == "__main__":
    main()