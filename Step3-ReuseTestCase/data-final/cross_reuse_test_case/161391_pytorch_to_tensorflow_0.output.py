import torch
import tensorflow as tf
import unittest
import gc
import psutil
import os

def mem_mb_print():
    gc.collect()
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    # Attempt to count TensorFlow objects held by Python GC
    tf_objs = sum(1 for o in gc.get_objects() if isinstance(o, (tf.Tensor, tf.Variable)))
    print(f"process: {mem // 1024**1} KB, TF objects: {tf_objs}")

class TestLeafGradTriu(unittest.TestCase):
    def test_cross_grad_triu(self):
        # Check for GPU availability to match original test's intent
        gpus = tf.config.list_physical_devices('GPU')
        device = "/GPU:0" if gpus else "/CPU:0"
        
        with tf.device(device):
            # Leaf tensor (Variable in TF)
            leaf = tf.Variable(tf.random.normal((8, 1, 24)), trainable=True)
            
            # Non-leaf tensor equivalent: A Variable derived from leaf to allow in-place ops
            # In PyTorch, no_leaf is a view. In TF, we use a Variable to hold the state.
            no_leaf = tf.Variable(leaf * 1.0, trainable=True)

            for _ in range(100):
                # Original logic: window_left = no_leaf[:, :, :-1]
                # Original logic: no_leaf[:, :, 1:] = torch.max(no_leaf[:, :, 1:], window_left)
                
                # Adapted logic using tf.experimental.numpy.triu
                # triu returns the upper triangular part of the array.
                # We assign it back to no_leaf to mimic the in-place operation.
                triu_val = tf.experimental.numpy.triu(no_leaf)
                no_leaf.assign(triu_val)
                
                # Python arithmetic operators
                mask = (no_leaf >= 0)
                mem_mb_print()

            loss = tf.reduce_sum(mask)
            
            # Verify gradients can be computed (backward graph check)
            with tf.GradientTape() as tape:
                # Re-compute loss inside tape context to ensure graph is traced
                # Note: In eager mode, the loop above already executed.
                # To check for leaks in graph building, one might use @tf.function,
                # but to match the original script's eager execution style:
                current_val = no_leaf * 1.0
                current_loss = tf.reduce_sum(current_val)
            
            grads = tape.gradient(current_loss, leaf)
            self.assertIsNotNone(grads)

if __name__ == "__main__":
    unittest.main()