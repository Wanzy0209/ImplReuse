import tensorflow as tf
import unittest
import gc
import psutil

def mem_mb_print():
    gc.collect()
    process = psutil.Process().memory_info().rss
    print(f"Process Memory: {process // 1024**2} MB")

class TestGeomspaceMemory(unittest.TestCase):
    @unittest.skipIf(len(tf.config.list_physical_devices('GPU')) == 0, "GPU not available")
    def test_geomspace_memory(self):
        # Create variables (leaf tensor equivalent)
        start = tf.Variable(1.0, dtype=tf.float32)
        stop = tf.Variable(10.0, dtype=tf.float32)

        for _ in range(100):
            with tf.GradientTape() as tape:
                # Call the API under test
                # geomspace generates a sequence, analogous to the operation in the loop
                res = tf.experimental.numpy.geomspace(start, stop, num=50)

                # Python arithmetic operators (similar to original)
                mask = (res >= 0)

                loss = tf.reduce_sum(mask)

            # Compute gradients to ensure graph execution
            grads = tape.gradient(loss, [start, stop])
            
            mem_mb_print()

if __name__ == "__main__":
    unittest.main()