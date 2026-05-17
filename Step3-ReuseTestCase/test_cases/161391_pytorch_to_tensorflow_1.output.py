import tensorflow as tf
import unittest
import gc
import psutil

def mem_mb_print():
    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %,"
          f"TF tensors: {sum(1 for o in gc.get_objects() if isinstance(o, tf.Tensor))}")

class TestTriuMemory(unittest.TestCase):
    @unittest.skipIf(not tf.config.list_physical_devices('GPU'), "CUDA not available")
    def test_triu_memory(self):
        device = "/GPU:0"

        with tf.device(device):
            # leaf GPU tensor (Variable with trainable=True mimics requires_grad=True)
            leaf = tf.Variable(tf.random.normal((8, 1, 24)), trainable=True)
            
            # Use GradientTape to track operations
            with tf.GradientTape() as tape:
                tape.watch(leaf)
                
                # non-leaf tensor
                no_leaf = leaf * 1.0

                for _ in range(100):
                    # Apply tf.keras.ops.triu
                    # Note: triu is not in-place, so we reassign to simulate the update
                    no_leaf = tf.keras.ops.triu(no_leaf)
                    
                    # Python arithmetic operators
                    mask = (no_leaf >= 0)
                    mem_mb_print()

            loss = tf.reduce_sum(mask)
            grads = tape.gradient(loss, leaf)
            self.assertIsNotNone(grads)

if __name__ == "__main__":
    unittest.main()