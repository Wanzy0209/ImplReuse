import torch
import tensorflow as tf

# Adapted test case for tf.compat.v1.train.string_input_producer
# The original bug report involved torch.compile failing with combo_kernels enabled.
# Since tf.compat.v1.train.string_input_producer is a data input pipeline utility
# and does not have a "combo_kernels" compilation step, we adapt the test to
# verify the API's behavior with multiple inputs and configurations (shuffle/epochs).

# Note: tf.compat.v1 APIs operate in graph mode.
tf.compat.v1.disable_eager_execution()

# Define inputs
# Original test used 3 different tensors. Here we use a list of strings.
input_strings = tf.constant(["file_0", "file_1", "file_2", "file_3", "file_4"])

# API Call
# Mimicking the complexity of the original test by enabling shuffle and epochs.
batch = tf.compat.v1.train.string_input_producer(
    input_strings,
    num_epochs=1,
    shuffle=True,
    capacity=32,
    seed=42
)

# Execution
with tf.compat.v1.Session() as sess:
    # Initialization is required for local variables (epochs counter)
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    # Coordinator and QueueRunners are necessary to drive the input pipeline
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # Original test executed the function once. Here we dequeue to verify the producer.
        # We run a few times to ensure the pipeline is active.
        results = []
        for _ in range(5):
            result = sess.run(batch)
            results.append(result)
            # Verify output type (TF 1.x strings are bytes)
            assert isinstance(result, bytes)
        
        print("Test passed. Successfully dequeued items:", results)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        coord.request_stop()
        coord.join(threads)