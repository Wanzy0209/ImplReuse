import tensorflow as tf
import numpy as np

# Note: QueueRunners are part of the TensorFlow 1.x graph execution mode.
# We must disable eager execution to use this API.
tf.compat.v1.disable_eager_execution()

def test_add_queue_runner():
    """
    Test case for tf.compat.v1.train.add_queue_runner.
    
    This test adapts the structure of the original PyTorch bug report:
    1. Setup inputs (Queue and Enqueue ops).
    2. Execute the target API (add_queue_runner).
    3. Run the graph and verify behavior (Session execution and assertions).
    """
    with tf.Graph().as_default():
        # 1. Setup: Define a FIFO queue and an enqueue operation.
        # This is analogous to defining the tensors q, k, v in the original PyTorch test.
        queue = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[()])
        enqueue_op = queue.enqueue(1.0)
        
        # Create a QueueRunner to manage the enqueue operation
        qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op])

        # 2. API Under Test: Add the QueueRunner to the graph's default collection.
        # This is the specific API requested to be tested.
        tf.compat.v1.train.add_queue_runner(qr)

        # Verification: Ensure the runner was added to the correct collection.
        collection = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
        assert qr in collection, "QueueRunner was not successfully added to the collection."

        # 3. Execution: Run the graph to verify the pipeline works.
        # This is analogous to the inductor(q, k, v) call and backward pass.
        with tf.compat.v1.Session() as sess:
            # Initialize local variables and start the queue runners
            sess.run(tf.compat.v1.local_variables_initializer())
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

            try:
                # Dequeue the item to verify the queue runner successfully enqueued data
                result = sess.run(queue.dequeue())
                
                # Assertion: Verify the data matches the expected value
                assert np.isclose(result, 1.0), f"Expected 1.0, but got {result}"
                print("Test passed: QueueRunner added and executed successfully.")
                
            finally:
                # Cleanup: Stop threads
                coord.request_stop()
                coord.join(threads)

if __name__ == "__main__":
    test_add_queue_runner()