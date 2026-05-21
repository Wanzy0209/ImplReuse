import unittest
import tensorflow as tf
import threading
import time

class TestQueueRunnerTimeout(unittest.TestCase):
    """
    Adapted from test_dtensor_compile_redistribute.
    
    The original test failed due to a subprocess timeout when executing
    a compiled graph with distributed tensors. This test adapts the logic
    to the TensorFlow API tf.compat.v1.train.add_queue_runner, verifying
    that the graph execution involving queue runners completes within a
    reasonable time limit, avoiding the deadlock/hang behavior observed
    in the original bug.
    """

    def test_add_queue_runner_no_timeout(self):
        # QueueRunners are only supported in graph mode (TF 1.x style)
        tf.compat.v1.disable_eager_execution()

        def execute_graph_logic():
            with tf.compat.v1.Session() as sess:
                # 1. Create a Queue
                queue = tf.compat.v1.FIFOQueue(capacity=10, dtypes=tf.float32, shapes=[])

                # 2. Define an enqueue operation
                enqueue_op = queue.enqueue([1.0])

                # 3. Create a QueueRunner
                # This creates threads that will run the enqueue_op
                qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op] * 2)

                # 4. Add the QueueRunner to the graph collection using the API under test
                tf.compat.v1.train.add_queue_runner(qr)

                # 5. Define a dequeue operation to consume data
                dequeue_op = queue.dequeue()

                # 6. Initialize variables and start the queue runners
                sess.run(tf.compat.v1.local_variables_initializer())
                
                # Coordinator for managing threads
                coord = tf.compat.v1.train.Coordinator()
                
                # Start the queue runners (this looks for the runner added via add_queue_runner)
                threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    # 7. Perform some work (dequeue items)
                    # If the queue runner hangs or deadlocks, this loop will not complete
                    for _ in range(5):
                        val = sess.run(dequeue_op)
                        self.assertEqual(val, 1.0)
                finally:
                    # Ensure clean shutdown
                    coord.request_stop()
                    coord.join(threads)

        # Helper to run the logic with a timeout, mimicking the original test's failure detection
        def run_with_timeout(target, timeout=10):
            exception = []
            result = []

            def worker():
                try:
                    result.append(target())
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                # If the thread is still alive, we have a timeout/hang similar to the original bug
                raise TimeoutError(f"Graph execution timed out after {timeout} seconds (Potential Deadlock)")
            
            if exception:
                raise exception[0]
            
            return result[0] if result else None

        # Execute the test with a strict timeout to catch hangs
        run_with_timeout(execute_graph_logic, timeout=5)

if __name__ == '__main__':
    unittest.main()