import torch
import tensorflow as tf
import numpy as np

def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention simulation with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Create captured buffer that depends on dynamic H
    # In TensorFlow, we use a Tensor or Variable. 
    head_scale = tf.random.normal([H], dtype=dtype)

    def score_mod(score, head_scale):
        # Simulate the score_mod logic: score * head_scale[head]
        # score shape: (B, H, S, S)
        # head_scale shape: (H,)
        # Reshape head_scale to (1, H, 1, 1) for broadcasting
        H_dim = tf.shape(score)[1]
        broadcast_scale = tf.reshape(head_scale, [1, H_dim, 1, 1])
        return score * broadcast_scale

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        with tf.device(device):
            q = tf.random.normal([B, H, S, D], dtype=dtype)
            k = tf.random.normal([B, H, S, D], dtype=dtype)
            v = tf.random.normal([B, H, S, D], dtype=dtype)

            # Use GradientTape to mimic requires_grad=True and backward()
            with tf.GradientTape() as tape:
                tape.watch([q, k, v, head_scale])
                
                # Call the compiled function
                outputs = compiled_fa(q, k, v, head_scale, score_mod)
                loss = tf.reduce_sum(outputs)

            grads = tape.gradient(loss, [q, k, v, head_scale])
            
            # Verify gradients are computed (mimicking successful backward pass)
            assert grads[0] is not None

    print(f"   Completed {i+1} iterations")


def main():
    # Determine device
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    dtype = tf.float16
    tf.random.set_seed(0)

    # Test with different head counts - this makes H a dynamic dimension
    # and the captured buffer (head_scale) changes size with H
    head_counts = [4, 8, 4, 16, 4]

    # Define the compiled function using tf.function (semantics of torch.compile)
    # We use experimental_relax_shapes=True to handle dynamic shapes (similar to dynamic=True)
    @tf.function(experimental_relax_shapes=True)
    def flex_attention_sim(q, k, v, head_scale, score_mod_fn):
        # Use the requested API: tf.keras.backend.name_scope
        # This scopes the operations within the graph, similar to how ops are organized in compiled models.
        with tf.keras.backend.name_scope("flex_attention"):
            # Simplified Attention Logic
            # 1. Calculate scores (Q * K^T)
            # q: (B, H, S, D), k: (B, H, S, D) -> scores: (B, H, S, S)
            scores = tf.matmul(q, k, transpose_b=True)
            
            # 2. Apply score_mod (capturing head_scale)
            scores = score_mod_fn(scores, head_scale)
            
            # 3. Softmax
            attn_weights = tf.nn.softmax(scores)
            
            # 4. Multiply by V
            # attn_weights: (B, H, S, S), v: (B, H, S, D) -> output: (B, H, S, D)
            output = tf.matmul(attn_weights, v)
            
            return output

    compiled_fa = flex_attention_sim

    print(f"Running flex-attention simulation with dynamic head counts on {device}, dtype={dtype}")
    print(f"Testing head counts: {head_counts}\n")

    for iteration, H in enumerate(head_counts, start=1):
        print(f"Iteration {iteration}:")
        run_with_head_count(compiled_fa, H, device, dtype)


if __name__ == "__main__":
    main()