import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import unittest
import gc
import psutil
import sys

def mem_mb_print():
    gc.collect()
    process = psutil.Process().memory_info().rss
    # Counting TF tensors is not as straightforward as torch.is_tensor(o) in gc.get_objects()
    # due to TF's object model, so we stick to process memory.
    print(f"process: {process // 1024**2} MB")

class TestVdotMemory(unittest.TestCase):
    def test_vdot_memory(self):
        # Setup leaf tensor (Variable with trainable=True)
        # Using CPU to ensure it runs everywhere, similar to the original test's intent
        leaf = tf.Variable(tf.random.normal((8, 1, 24)), trainable=True)
        
        # Create a non-leaf tensor (result of an operation)
        no_leaf = leaf * 1.0

        for _ in range(100):
            # Slicing logic similar to the original test
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            window_right = no_leaf[:, :, 1:]         # [8, 1, 23]

            # Use the similar API: tf.experimental.numpy.vdot
            # Note: vdot is a reduction operation (returns a scalar), unlike torch.max which can be element-wise.
            # Therefore, we cannot perform an in-place assignment like the original PyTorch bug.
            # We execute the operation to check for memory leaks in a gradient context.
            res = tnp.vdot(window_right, window_left)
            
            # Python arithmetic operators
            mask = (res >= 0)
            
            mem_mb_print()

if __name__ == "__main__":
    unittest.main()