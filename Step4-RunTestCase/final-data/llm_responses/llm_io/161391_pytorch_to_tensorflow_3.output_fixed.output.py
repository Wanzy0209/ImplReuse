import tensorflow as tf
import unittest
import gc

# Handle missing psutil dependency gracefully
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def mem_mb_print():
    gc.collect()
    if HAS_PSUTIL:
        process = psutil.Process().memory_info().rss
        # Counting TF tensors in gc is not straightforward as PyTorch, 
        # but we can monitor process memory.
        print(f"Process Memory: {process // 1024**2} MB")
    else:
        print("Process Memory: psutil not available, skipping check.")

class TestTrilMemory(unittest.TestCase):
    def test_tril_memory(self):
        # Create a leaf tensor (Variable) with gradient tracking
        # Shape (8, 1, 24) is valid for tril (operates on last 2 dims)
        leaf = tf.Variable(tf.random.normal((8, 1, 24)), trainable=True)
        no_leaf = leaf * 1.0

        for _ in range(100):
            # Apply tf.keras.ops.tril
            # Note: tril is not an in-place operation in TF, it returns a new tensor.
            # We assign it back to no_leaf to simulate the update cycle.
            no_leaf = tf.keras.ops.tril(no_leaf)

            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        # Ensure gradients are computed to verify autograd context
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(mask)
        grads = tape.gradient(loss, leaf)
        self.assertIsNotNone(grads)

if __name__ == "__main__":
    unittest.main()