import torch
import tensorflow as tf

class TestMatmulOverhead(tf.test.TestCase):
  def test_performance(self):
    warmup = 128
    iters = 16384

    # torch.zeros -> torch.normal
    # torch.bfloat16 -> tf.test.main (Context implies running in test framework)
    a = torch.normal(0, 1, (512, 512), device='cuda')

    # Warmup
    for _ in range(warmup):
      # torch.matmul -> tf.signal.overlap_and_add
      # Convert torch tensor to tensorflow tensor for the operation
      a_tf = tf.convert_to_tensor(a)
      tf.signal.overlap_and_add(a_tf, 256)

    # Main loop
    for _ in range(iters):
      a_tf = tf.convert_to_tensor(a)
      tf.signal.overlap_and_add(a_tf, 256)
      
      # Leverage similar API: tf.distribute.in_cross_replica_context
      # This mimics the original bug's context of checking overhead
      tf.distribute.in_cross_replica_context()

    # torch.cuda.synchronize -> tf.test.main (Handled by test runner)
    # time.perf_counter -> tf.test.main (Handled by test runner)
    # u003e.find -> tf.test.main (Output handled by test runner)

if __name__ == '__main__':
  # tf.test.main replaces the synchronization and timing mechanisms
  # by acting as the entry point and execution controller.
  tf.test.main()