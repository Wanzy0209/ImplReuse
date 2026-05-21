import torch
import sys
import tensorflow as tf

# Preserve the recursion limit setting from the original bug report
# to verify if the API respects or ignores system limits (if applicable).
sys.setrecursionlimit(10000000)

# Adapt the 'outer' function to use the TensorFlow API.
# The original 'outer' used torch.compile on a recursive function.
# Here, we use tf.compat.v1.train.range_input_producer to generate a large range,
# analogous to the deep recursion in the original test.
def outer(limit):
    return tf.compat.v1.train.range_input_producer(
        limit=limit, 
        capacity=limit + 10, 
        shuffle=False
    )

# Setup TensorFlow v1 session (required for compat.v1 APIs)
with tf.compat.v1.Session() as sess:
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Call the API with a large limit (analogous to n=1000 in the original).
    # The original bug caused a RecursionError because the C-level limit 
    # was not updated by sys.setrecursionlimit.
    # We verify that the TensorFlow API handles the large limit correctly.
    queue = outer(1000)
    
    # Dequeue an element to trigger the execution logic
    result = sess.run(queue.dequeue())
    
    # Assertion to verify the behavior is as expected (no crash, correct output)
    assert result == 0, "Expected the first element of the range to be 0"