import torch
import tensorflow as tf
import numpy as np

# Ensure we are using TensorFlow v1 compatibility mode for this specific API
# Note: string_input_producer is a v1 API and does not support eager execution.
# We disable eager execution to simulate the graph environment.
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# Replicate the seed setup from the original test
tf.random.set_seed(19989)
np.random.seed(19989)

def fuzzed_program(arg_0, sentinel):
    """
    Adapted function using tf.compat.v1.train.string_input_producer.
    Original PyTorch logic performed arithmetic operations.
    TensorFlow logic performs queue operations.
    """
    # arg_0 is expected to be a list of strings for this API
    # sentinel is kept for signature compatibility with the original test
    string_tensor = tf.convert_to_tensor(arg_0, dtype=tf.string)

    # The API under test: string_input_producer
    # This creates a queue and enqueues the strings.
    # It is analogous to setting up a data pipeline node in the graph.
    queue = tf.compat.v1.train.string_input_producer(
        string_tensor,
        num_epochs=None,
        shuffle=False,
        seed=19989,
        capacity=32
    )

    # Dequeue the next string (analogous to getting the result)
    result = queue.dequeue()
    return result

# Setup inputs
# Original: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()
# Adapted: Generate a random list of strings to simulate dynamic input
num_strings = np.random.randint(1, 10)
arg_0 = [f"file_{i}.dat" for i in range(num_strings)]

# Sentinel tensor (unused in TF logic but kept for structure)
sentinel = tf.constant(1.0)

args = (arg_0, sentinel)

# Execution
# Since string_input_producer is a graph-based API, we run it inside a Session.
# This corresponds to the "compiled" execution path in the original PyTorch test.
with tf.compat.v1.Session() as sess:
    # Initialize local variables (required for epochs counter)
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    # Start QueueRunners to execute the queue operations
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

    try:
        # Build the graph (Compilation phase)
        result_node = fuzzed_program(*args)
        
        # Run the graph (Execution phase)
        # We run it a few times to verify the queue cycles correctly
        for _ in range(3):
            result_val = sess.run(result_node)
            print(f' success: {result_val.decode()}')
            
    except Exception as e:
        print(f' Error: {e}')
    finally:
        # Stop the queue runners
        coord.request_stop()
        coord.join(threads)