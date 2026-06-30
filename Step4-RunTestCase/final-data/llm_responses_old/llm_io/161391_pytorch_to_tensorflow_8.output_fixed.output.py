import torch
import unittest
import gc
import psutil

def mem_mb_print():
    gc.collect()
    process = psutil.Process().memory_info().rss
    # Using MB for readability, similar to original logic
    print(f"Process RSS: {process // 1024**2} MB")

class TestFullLikeMemory(unittest.TestCase):
    def test_cross_grad_full_like(self):
        try:
            import tensorflow as tf
        except ImportError as e:
            self.skipTest(f"TensorFlow import failed (environment issue): {e}")
        
        if not tf.config.list_physical_devices('GPU'):
            self.skipTest("CUDA not available")
        
        # TensorFlow equivalent of leaf GPU tensor with requires_grad=True
        # We use a tf.Variable which is tracked by default for gradients
        leaf = tf.Variable(tf.random.normal((8, 1, 24)), dtype=tf.float32)
        
        # In TensorFlow, tensors are immutable. The concept of a "non-leaf" tensor 
        # that is modified in-place is handled by updating a tf.Variable.
        
        for _ in range(100):
            window_left = leaf[:, :, :-1]  # [8, 1, 23]
            
            # Original PyTorch logic: no_leaf[:, :, 1:] = torch.max(no_leaf[:, :, 1:], window_left)
            # Adaptation: We use the Similar API (tf.keras.ops.full_like) to generate the update tensor.
            # To maintain a connection to the original logic, we calculate a fill value based on window_left.
            fill_value = tf.reduce_max(window_left)
            
            # Use tf.keras.ops.full_like to create the tensor for the update
            # This creates a tensor of shape [8, 1, 23] filled with the max value
            update_tensor = tf.keras.ops.full_like(leaf[:, :, 1:], fill_value)
            
            # Mimic the slice assignment: no_leaf[:, :, 1:] = update_tensor
            # Since TF does not support direct slice assignment on tensors, we reconstruct 
            # the tensor and assign it back to the variable to simulate the in-place operation.
            left_slice = leaf[:, :, :1] # [8, 1, 1]
            updated_tensor = tf.concat([left_slice, update_tensor], axis=2)
            
            # Perform the "in-place" update
            leaf.assign(updated_tensor)
            
            # Python arithmetic operators (equivalent to mask = (no_leaf >= 0))
            mask = (leaf >= 0)
            mem_mb_print()

        # Ensure operations are executed
        loss = tf.reduce_sum(mask)

if __name__ == "__main__":
    unittest.main()