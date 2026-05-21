import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use TF 1.x graph mode (analogous to compilation)
tf.compat.v1.disable_eager_execution()

def f(x, max_val):
    """
    Adapted function mimicking the original PyTorch logic.
    Original: y = torch.clamp(x, 0, max_val.item())
    Adapted: Use the scalar tensor 'max_val' to control the input producer.
    """
    # In the original bug, max_val.item() is used to extract a scalar for control flow.
    # Here, we pass the tensor 'max_val' directly to num_epochs to test 
    # how the API handles tensor arguments for scalar parameters.
    queue = tf.compat.v1.train.string_input_producer(
        x,
        num_epochs=max_val,
        shuffle=False
    )
    return queue

# Setup inputs
# Mimic x = torch.randn(...) -> A tensor of strings
filenames = tf.constant(["file1.txt", "file2.txt", "file3.txt"])
# Mimic max_val = torch.tensor(5.0) -> A scalar tensor (int for epochs)
max_val = tf.constant(2)

# Build the graph (analogous to torch.compile)
queue = f(filenames, max_val)

# Execution
with tf.compat.v1.Session() as sess:
    # Initialize local variables (required for num_epochs counter in string_input_producer)
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    results = []
    try:
        # We expect 3 files * 2 epochs = 6 items
        for _ in range(6):
            result = sess.run(queue.dequeue())
            results.append(result)
            print(f"Dequeued: {result}")
    except tf.errors.OutOfRangeError:
        print("Queue exhausted (expected behavior).")
    finally:
        coord.request_stop()
        coord.join(threads)

    # Verify behavior
    assert len(results) == 6, f"Expected 6 results (3 files * 2 epochs), but got {len(results)}"
    print("Test passed: API handled tensor argument correctly.")