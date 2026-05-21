import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use v1 APIs like string_input_producer effectively
# This mimics the "compiled" graph environment where the original bug occurred.
tf.compat.v1.disable_eager_execution()

def test_string_input_producer():
    # Define input data (Strings instead of Sparse Tensors)
    # Adapting the input to fit the API's requirements
    filenames = ["file1.txt", "file2.txt", "file3.txt"]
    string_tensor = tf.convert_to_tensor(filenames, dtype=tf.string)

    # Define the model/pipeline using the similar API
    # Analogous to the PyTorch model definition
    def input_pipeline():
        # API Under Test: tf.compat.v1.train.string_input_producer
        queue = tf.compat.v1.train.string_input_producer(
            string_tensor, 
            num_epochs=1, 
            shuffle=False,
            capacity=10
        )
        return queue.dequeue()

    # Get the output tensor
    output = input_pipeline()

    # Initialize variables and local variables (for epochs)
    init_op = tf.group(
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()
    )

    print("Running input pipeline...")

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        
        # Start queue runners (required for string_input_producer)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            # Run the pipeline and verify outputs
            # PyTorch equivalent: print("Compiled output:", torch.compile(model)(x))
            print("Output 1:", sess.run(output).decode())
            print("Output 2:", sess.run(output).decode())
            print("Output 3:", sess.run(output).decode())

            # Verify behavior: should raise OutOfRangeError after num_epochs
            try:
                sess.run(output)
                assert False, "Expected OutOfRangeError"
            except tf.errors.OutOfRangeError:
                print("Successfully caught OutOfRangeError as expected.")

        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_string_input_producer()