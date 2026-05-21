import torch
import tensorflow as tf

def test_add_queue_runner_global_state():
    """
    Adapted test case based on PyTorch Issue 167064.
    
    The original bug describes a scenario where calling torch.compile
    inadvertently modifies global state (torch.distributions settings).
    
    This test verifies the behavior of the similar TensorFlow API
    tf.compat.v1.train.add_queue_runner, checking if it modifies
    the global graph collection state as expected (or unexpectedly).
    """
    # Disable eager execution to use graph mode (required for compat.v1 APIs)
    tf.compat.v1.disable_eager_execution()

    # Create a new graph context to isolate the test
    with tf.compat.v1.Graph().as_default():
        # Setup: Create a simple Queue and QueueRunner
        queue = tf.compat.v1.queue.FIFOQueue(10, tf.float32)
        enqueue_op = queue.enqueue([1.0])
        qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

        # Check initial state of the global collection
        # Analogous to checking torch.distributions settings before compile
        initial_runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
        assert len(initial_runners) == 0, "Collection should be empty before API call"

        # Call the API under test
        # Analogous to calling torch.compile
        tf.compat.v1.train.add_queue_runner(qr)

        # Verify the side effect on global state
        # Analogous to checking if torch.distributions settings changed
        final_runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
        assert len(final_runners) == 1, "Collection should contain one runner after API call"
        assert final_runners[0] == qr, "The runner in the collection should match the one added"

if __name__ == "__main__":
    test_add_queue_runner_global_state()
    print("Test passed: API behavior verified.")