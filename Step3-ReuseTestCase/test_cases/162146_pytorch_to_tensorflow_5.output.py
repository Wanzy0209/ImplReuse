import torch
import tensorflow as tf

# Note: tf.compat.v1.train.add_queue_runner is designed for TF1 graph mode.
# We must disable eager execution to use this API.
tf.compat.v1.disable_eager_execution()

def test_add_queue_runner():
    """
    Adapted test case for tf.compat.v1.train.add_queue_runner.
    
    Original PyTorch Logic:
    1. Define a function with side effects (in-place ops).
    2. Execute in eager mode.
    3. Execute in compiled mode.
    4. Assert results match.
    
    Adapted TensorFlow Logic:
    1. Define a QueueRunner (the object to be added).
    2. Add it to the graph collection using the API.
    3. Verify the side effect (the runner is present in the collection).
    """
    
    # Reset default graph to ensure a clean state for the test
    tf.compat.v1.reset_default_graph()

    # Setup: Create a simple FIFO queue and a QueueRunner
    # This mimics the setup of tensors in the original test
    queue = tf.compat.v1.queue.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[])
    enqueue_op = queue.enqueue([1.0])
    qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

    # Action: Use the API under test to add the runner to the default collection
    # This corresponds to the 'cfoo = torch.compile(foo)' and execution phase
    tf.compat.v1.train.add_queue_runner(qr)

    # Verification: Check if the runner was successfully added to the collection
    # This corresponds to 'torch.testing.assert_close(res, cres)'
    collected_runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
    
    assert len(collected_runners) == 1, f"Expected 1 runner in collection, found {len(collected_runners)}"
    assert collected_runners[0] is qr, "The QueueRunner in the collection is not the one we added."

    print("Test passed: tf.compat.v1.train.add_queue_runner successfully added the runner to the collection.")

if __name__ == "__main__":
    test_add_queue_runner()