import tensorflow as tf
import unittest
import gc
import tensorflow.experimental.numpy as tnp

# Handle missing psutil gracefully
try:
    import psutil
except ImportError:
    psutil = None

def mem_mb_print():
    if psutil is None:
        print("psutil not installed, skipping memory stats")
        return

    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %")

class TestLeafGradGcd(unittest.TestCase):
    def test_cross_grad_gcd(self):
        # leaf tensor (Variable in TF)
        # Using GPU if available, similar to the original test's intent
        device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        with tf.device(device):
            leaf = tf.Variable(tf.random.normal((8, 1, 24)), trainable=True)
            
            # non-leaf equivalent. 
            # In PyTorch, this is a tensor with a grad_fn. In TF, to allow in-place updates 
            # (assignment), we must use a tf.Variable. We initialize it based on the leaf.
            no_leaf = tf.Variable(leaf * 1.0)

            for _ in range(100):
                window_left = no_leaf[:, :, :-1]  # [8, 1, 23]
                
                # inplace operators simulation
                # Using tf.experimental.numpy.gcd as the similar API
                # Note: TF tensors are immutable, so we use assign on the Variable slice
                res = tnp.gcd(no_leaf[:, :, 1:], window_left)
                no_leaf[:, :, 1:].assign(res)
                
                # Python arithmetic operators
                mask = (no_leaf >= 0)
                mem_mb_print()

            # Verify gradients can be computed (sanity check)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(mask)
            grad = tape.gradient(loss, leaf)
            self.assertIsNotNone(grad)

if __name__ == "__main__":
    unittest.main()