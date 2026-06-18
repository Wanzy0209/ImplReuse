import torch
import tensorflow as tf
import numpy as np

# Adapted from PyTorch's flex_attention test case to TensorFlow.
# The original bug involves dynamic buffers (head_scale) changing size 
# within a compiled context (torch.compile).
# Here we test tf.keras.name_scope within a tf.function (compiled context)
# to ensure it handles dynamic shape retracing correctly.

def run_with_head_count(compiled_fa, H, device, dtype):
    """Run the compiled function with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Create captured buffer that depends on dynamic H
    # In PyTorch: head_scale = torch.randn(H, ...)
    head_scale = tf.random.normal([H], dtype=dtype)

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        # PyTorch: q = torch.randn(B, H, S, D, ...)
        q = tf.random.normal([B, H, S, D], dtype=dtype)
        k = tf.random.normal([B, H, S, D], dtype=dtype)
        v = tf.random.normal([B, H, S, D], dtype=dtype)

        with tf.GradientTape() as tape:
            # Watch inputs for gradients (mimics requires_grad=True)
            tape.watch([q, k, v, head_scale])

            # Run the compiled function
            # In PyTorch: outputs = compiled_fa(q, k, v, score_mod=..., block_mask=...)
            # Here we pass head_scale directly to simulate the captured buffer behavior
            outputs = compiled_fa(q, k, v, head_scale)
            
            loss = tf.reduce_sum(outputs)

        # Calculate gradients (mimics loss.backward())
        grads = tape.gradient(loss, [q, k, v, head_scale])
        
        # Basic assertion to ensure gradients are computed and shapes match
        assert grads[0] is not None
        assert grads[0].shape == q.shape

    print(f"   Completed {i+1} iterations")


def main():
    # Determine device (GPU or CPU)
    device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
    dtype = tf.float16
    
    # Set seed for reproducibility
    tf.random.set_seed(0)

    # Test with different head counts - this makes H a dynamic dimension
    # and the captured buffer (head_scale) changes size with H
    head_counts = [4, 8, 4, 16, 4]

    # Define the function to be compiled
    # Mimics torch.compile(flex_attention, ...)
    @tf.function(experimental_recompile=5) # Allow some recompilations for dynamic shapes
    def flex_attention_sim(q, k, v, head_scale):
        # Use tf.keras.name_scope as the context manager under test
        with tf.keras.name_scope("flex_attention"):
            # Reshape head_scale to broadcast across the batch, sequence, and head dimensions
            # head_scale shape: [H]
            # Target shape: [1, H, 1, 1]
            scale = tf.reshape(head_scale, [1, tf.shape(head_scale)[0], 1, 1])
            
            # Apply scaling (mimics score_mod logic)
            scaled_q = q * scale
            
            # Simplified attention mechanism (MatMul)
            # Transpose k: [B, H, S, D] -> [B, H, D, S]
            kt = tf.transpose(k, [0, 1, 3, 2])
            
            # Attention scores: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
            attn_scores = tf.matmul(scaled_q, kt)
            
            # Output: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
            outputs = tf.matmul(attn_scores, v)
            
            return outputs

    print(f"Running flex-attention simulation with dynamic head counts on {device}, dtype={dtype}")
    print(f"Testing head counts: {head_counts}\n")

    with tf.device(device):
        for iteration, H in enumerate(head_counts, start=1):
            print(f"Iteration {iteration}:")
            run_with_head_count(flex_attention_sim, H, device, dtype)


if __name__ == "__main__":
    main()