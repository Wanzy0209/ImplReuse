import torch
import tensorflow as tf
import gc

def get_tensor_count():
    """
    Helper function to count the number of live tf.Tensor objects.
    This mimics the 'Tensors: X' metric in the original PyTorch bug report.
    """
    return sum(1 for obj in gc.get_objects() if isinstance(obj, tf.Tensor))

def test_name_scope_memory_leak():
    """
    Adapted from PyTorch Issue 165407.
    
    Original Bug: torch.compile memory leak involving flash_attn_varlen_func.
    Symptom: The number of tensors increases every step (e.g., 2266 -> 2602 -> 3034).
    
    This test verifies if using tf.keras.backend.name_scope (the similar API)
    in a repeated execution context causes a similar accumulation of tensors
    (memory leak).
    """
    print("Testing tf.keras.backend.name_scope for memory leaks...")
    print("Step | Tensors")
    print("-" * 20)

    # Simulating the training loop from the bug report
    # We iterate and check tensor counts at intervals
    for step in range(150, 351, 50):
        
        # Run a batch of operations
        for _ in range(50):
            # Use the API under test: tf.keras.backend.name_scope
            with tf.keras.backend.name_scope("attention_sim"):
                # Simulate operations similar to the attention mechanism
                # Creating random tensors to mimic input/activations
                q = tf.random.normal([32, 10, 64])
                k = tf.random.normal([32, 10, 64])
                
                # Perform a computation (MatMul as a proxy for attention scores)
                # This creates intermediate tensors that should be released
                attn_scores = tf.matmul(q, k, transpose_b=True)
                
                # Ensure the tensor is used to prevent dead-code elimination
                _ = tf.reduce_sum(attn_scores)

        # Force garbage collection to clear unreferenced objects
        gc.collect()
        
        # Check the number of live tensors
        current_tensor_count = get_tensor_count()
        
        # Print status similar to the original bug report
        print(f"Step {step} | Tensors: {current_tensor_count}")

        # Note: In the original bug, the count increased linearly (e.g., +300 every 50 steps).
        # A healthy implementation should show a stable or fluctuating count, 
        # not a continuous unbounded growth.

if __name__ == "__main__":
    test_name_scope_memory_leak()