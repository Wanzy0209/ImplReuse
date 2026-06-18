import tensorflow as tf

# The API tf.compat.v1.train.string_input_producer relies on graph mode and queues,
# which are not supported in eager execution. We must disable eager execution.
tf.compat.v1.disable_eager_execution()

def get_sample_inputs():
    """
    Generates a list of string inputs (e.g., filenames) to be processed.
    Corresponds to get_sample_inputs in the PyTorch test case.
    """
    return ["file_{}.jpg".format(i) for i in range(10)]

def main():
    # Setup inputs
    filenames = get_sample_inputs()
    string_tensor = tf.constant(filenames)
    
    print(f'Input tensor: {string_tensor}')

    # Call the API
    # This is analogous to the compiled model execution in the PyTorch example.
    # The API sets up a queue pipeline to output strings.
    output = tf.compat.v1.train.string_input_producer(string_tensor, shuffle=False)

    # Verify the output type and shape access
    # The PyTorch bug occurred because the output was a tuple, causing .shape to fail.
    # Here we verify if the TensorFlow API returns an object that supports .shape.
    print(f'Output type: {type(output)}')
    
    try:
        shape = output.shape
        print(f'Output shape: {shape}')
    except AttributeError as e:
        print(f'AttributeError: {e}')

    # To ensure the test is runnable and verifies the pipeline works, we run a session.
    with tf.compat.v1.Session() as sess:
        # Initialize local variables (not strictly needed if num_epochs is None, but good practice)
        sess.run(tf.compat.v1.local_variables_initializer())
        
        # Start the queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            # Dequeue and print a few values to confirm execution
            for _ in range(3):
                val = sess.run(output)
                print(f'Dequeued value: {val}')
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()