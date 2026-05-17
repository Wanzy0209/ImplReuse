import torch
import tensorflow as tf

# Ensure we are in graph mode as add_queue_runner is a TF1 graph concept
tf.compat.v1.disable_eager_execution()

def test_queue_runner_shape_handling():
    """
    Adapts the torch.compile regression test (Issue 161372) to TensorFlow.
    
    The original bug involved a tensor size mismatch (expected 77, actual 78) 
    causing torch.compile to hit a recompilation limit. 
    
    This test verifies that tf.compat.v1.train.add_queue_runner correctly handles 
    the registration of queue runners with different tensor shapes, ensuring no 
    regression in graph collection management when encountering dynamic or 
    mismatched dimensions.
    """
    with tf.compat.v1.Graph().as_default():
        # Scenario 1: Queue runner with size 77 (from the bug report)
        queue_77 = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[77])
        enqueue_77 = queue_77.enqueue([tf.zeros([77], dtype=tf.float32)])
        qr_77 = tf.compat.v1.train.QueueRunner(queue_77, [enqueue_77])

        # Add the first runner
        tf.compat.v1.train.add_queue_runner(qr_77)

        # Scenario 2: Queue runner with size 78 (the mismatched size from the bug report)
        queue_78 = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[78])
        enqueue_78 = queue_78.enqueue([tf.zeros([78], dtype=tf.float32)])
        qr_78 = tf.compat.v1.train.QueueRunner(queue_78, [enqueue_78])

        # Add the second runner
        tf.compat.v1.train.add_queue_runner(qr_78)

        # Verification: Check that both runners are successfully added to the collection
        # This mirrors the expectation that the system should handle the changing shapes
        # without crashing or hitting a limit.
        collected_runners = tf.compat.v1.train.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)

        assert len(collected_runners) == 2, \
            f"Expected 2 queue runners in collection, but found {len(collected_runners)}"
        assert collected_runners[0] is qr_77, "First queue runner mismatch"
        assert collected_runners[1] is qr_78, "Second queue runner mismatch"

        print("Test Passed: Queue runners with dynamic shapes handled correctly.")

if __name__ == "__main__":
    test_queue_runner_shape_handling()