import tensorflow as tf

# Ensure we are in TF1 graph mode as the API is compat.v1
tf.compat.v1.disable_eager_execution()

def build_and_check_graph(i):
    """
    Mimics the structure of the original PyTorch function 'fn'.
    In PyTorch, i==1 triggers a graph break.
    Here, i==1 triggers adding a queue runner to the graph.
    """
    with tf.compat.v1.Graph().as_default():
        # Setup basic queue operations
        q = tf.compat.v1.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[])
        enqueue_op = q.enqueue([1.0])
        qr = tf.compat.v1.train.QueueRunner(q, [enqueue_op])

        # Conditional logic based on input 'i'
        if i == 1:
            # This is the API under test
            tf.compat.v1.train.add_queue_runner(qr)

        # Verify the graph state
        # In the PyTorch bug, the graph became empty.
        # Here we check if the queue runner collection is populated as expected.
        runners = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)
        return len(runners)

# Test Case 1: i = 0 (No queue runner added)
count_0 = build_and_check_graph(0)
assert count_0 == 0, f"Expected 0 runners for i=0, got {count_0}"

# Test Case 2: i = 1 (Queue runner added - analogous to the graph break path)
count_1 = build_and_check_graph(1)
assert count_1 == 1, f"Expected 1 runner for i=1, got {count_1}"

# Test Case 3: i = 2 (No queue runner added)
count_2 = build_and_check_graph(2)
assert count_2 == 0, f"Expected 0 runners for i=2, got {count_2}"

print("Test passed: tf.compat.v1.train.add_queue_runner behaves correctly under conditional logic.")