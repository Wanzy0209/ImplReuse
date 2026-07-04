import unittest
import gc
import psutil

# Attempt to import dependencies to handle environment issues gracefully
try:
    import torch
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp
    # Enable numpy behavior to ensure tnp operations integrate with the TF graph
    tnp.experimental_enable_numpy_behavior()
    DEPENDENCIES_AVAILABLE = True
    IMPORT_ERROR_MSG = None
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)

def mem_mb_print():
    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %")

class TestBroadcastArraysMemory(unittest.TestCase):
    def setUp(self):
        # Skip test if dependencies failed to import due to environment issues (e.g., GLIBCXX)
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Skipping test due to missing dependencies or environment error: {IMPORT_ERROR_MSG}")

    def test_cross_grad_memory(self):
        # Create a leaf tensor (Variable)
        leaf = tf.Variable(tf.random.normal((8, 1, 24)), name='leaf')
        # Create a non-leaf tensor
        no_leaf = leaf * 1.0

        for _ in range(100):
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            
            # Use GradientTape to mimic requires_grad=True context
            with tf.GradientTape() as tape:
                tape.watch(no_leaf)
                
                # Call the similar API: tf.experimental.numpy.broadcast_arrays
                # Note: broadcast_arrays is not an in-place operation, unlike the original torch.max.
                # We adapt the logic to check memory usage during repeated calls.
                res = tnp.broadcast_arrays(no_leaf[:, :, 1:], window_left)
                
                # Python arithmetic operators
                mask = (res[0] >= 0)
            
            mem_mb_print()

if __name__ == "__main__":
    unittest.main()