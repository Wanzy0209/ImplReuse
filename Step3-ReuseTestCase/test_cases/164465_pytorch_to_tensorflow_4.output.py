import torch
import tensorflow as tf

def test_string_input_producer_int64_max():
    """
    Adapts the PyTorch bug reproduction logic (int64 range generation + view + max)
    to the TensorFlow API tf.compat.v1.train.string_input_producer.
    
    The original bug involves:
    1. Generating a range of integers (iota).
    2. Viewing/Reshaping the tensor.
    3. Computing the max value.
    
    This test verifies that the string_input_producer can handle a sequence of inputs
    that represent this range, and that subsequent operations (conversion to int64,
    reshape, and max) execute correctly within the TF v1 graph context.
    """
    # Disable eager execution to use TF v1 graph mode (analogous to torch.compile context)
    tf.compat.v1.disable_eager_execution()

    # Parameters mirroring the PyTorch repro
    range_size = 36
    start = 0
    step = 1
    
    # Create a string tensor representing the iota sequence (0 to 35)
    # string_input_producer requires string input
    string_sequence = [str(i) for i in range(start, range_size, step)]
    input_tensor = tf.constant(string_sequence, dtype=tf.string)

    # Use the similar API: string_input_producer
    # This acts as the input source (similar to how iota generates input in the original)
    queue = tf.compat.v1.train.string_input_producer(
        input_tensor,
        num_epochs=1,
        shuffle=False,  # Keep order to match iota behavior
        capacity=64,    # Ensure capacity fits the range
        name="string_producer"
    )

    # To mimic the "view" and "max" on the whole tensor, we dequeue the whole batch.
    # string_input_producer outputs a single string per dequeue, so we batch them.
    batch = tf.train.batch(
        [queue],
        batch_size=range_size,
        capacity=range_size + 10,
        enqueue_many=False,
        name="batch_range"
    )

    # Convert strings to int64 (mimicking dtype=torch.int64 in the bug report)
    int_tensor = tf.strings.to_number(batch, out_type=tf.int64)

    # Mimic view_3: view(iota, [1, 36])
    reshaped_tensor = tf.reshape(int_tensor, [1, range_size])

    # Mimic max_1: max(view_3)
    max_val = tf.reduce_max(reshaped_tensor)

    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for num_epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Execute the graph
            result = sess.run(max_val)
            
            # Verify the result
            # The max of range(0, 36) is 35
            expected_max = range_size - 1
            assert result == expected_max, f"Expected max {expected_max}, but got {result}"
            print(f"Test Passed. Max value: {result}")
            
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_string_input_producer_int64_max()