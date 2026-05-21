import torch
import sys
import tensorflow as tf

# Disable eager execution to use the v1 API (string_input_producer) properly
tf.compat.v1.disable_eager_execution()

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1)

# Use tf.function to mimic the compilation aspect of torch.compile
@tf.function
def outer(x):
    return fn(x, 1000)

# Set up the input producer using the requested API
string_tensor = tf.constant(["file1.txt", "file2.txt"])
queue = tf.compat.v1.train.string_input_producer(string_tensor, num_epochs=1, shuffle=False)
dequeue_op = queue.dequeue()

# Set recursion limit high, as in the original bug report
sys.setrecursionlimit(10000000)

with tf.compat.v1.Session() as sess:
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    
    # Start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        # Get a string from the queue
        input_string = sess.run(dequeue_op)
        
        # Attempt to run the recursive function
        # Note: In TF graph mode, calling outer(input_string) builds the graph.
        # Since n=1000 is a Python int, this attempts to unroll the recursion 1000 times
        # during graph construction, testing the recursion limit.
        result = sess.run(outer(input_string))
        print("Test passed. Result:", result)
        
    except RecursionError as e:
        print(f"RecursionError encountered: {e}")
    finally:
        coord.request_stop()
        coord.join(threads)