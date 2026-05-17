import tensorflow as tf

# Note: QueueRunners are part of TensorFlow 1.x graph mode.
# We must disable eager execution to use this API.
tf.compat.v1.disable_eager_execution()

def test_add_queue_runner():
    """
    Test case for tf.compat.v1.train.add_queue_runner.
    
    This test verifies that a QueueRunner can be successfully added to the 
    default graph collection. While the original issue involved PyTorch Inductor 
    and Triton compilation, this test adapts the structural logic (setup -> 
    configuration -> execution -> verification) to the TensorFlow QueueRunner API.
    """
    
    # 1. Setup: Create a simple FIFO queue and an enqueue operation
    # This mimics the tensor setup in the original PyTorch test.
    queue_capacity = 10
    dtypes = [tf.float32]
    shapes = [()]
    
    queue = tf.compat.v1.FIFOQueue(capacity=queue_capacity, dtypes=dtypes, shapes=shapes)
    enqueue_op = queue.enqueue(1.0)

    # 2. Configuration: Create a QueueRunner
    # This mimics the compilation/definition step in the original test.
    # We create a runner with two threads attempting to enqueue the same value.
    qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op] * 2)

    # 3. Execution: Add the queue runner to the default collection
    # This is the API under test.
    tf.compat.v1.train.add_queue_runner(qr)

    # 4. Verification: Assert that the runner was added to the collection
    graph = tf.compat.v1.get_default_graph()
    collected_runners = graph.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)

    assert qr in collected_runners, "QueueRunner was not successfully added to the graph collection."
    assert len(collected_runners) == 1, "Expected exactly one QueueRunner in the collection."

    print("Test passed: tf.compat.v1.train.add_queue_runner behaves as expected.")

if __name__ == "__main__":
    test_add_queue_runner()