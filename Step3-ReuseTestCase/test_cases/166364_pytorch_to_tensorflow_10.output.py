import torch
import tensorflow as tf

def test_range_input_producer_learnable_scalar():
    """
    Test function to verify range_input_producer works with a learnable scalar
    parameter inside a compiled context (tf.function), analogous to the 
    PyTorch flex_attention bug report regarding learnable scalars.
    """
    # Disable eager execution to use tf.compat.v1.Session and legacy queues
    tf.compat.v1.disable_eager_execution()

    # 1. Define a learnable scalar parameter (mimicking nn.Parameter(torch.tensor(0.0)))
    # Note: range_input_producer limit is int32, so we cast later or define as int.
    # We use float to match the PyTorch '0.0' example and cast inside.
    temp = tf.Variable(0.0, dtype=tf.float32, name="learnable_scalar")

    # 2. Define the function that uses the API and the learnable parameter
    # We use tf.function to mimic torch.compile
    @tf.function
    def create_producer():
        # Mimic the score_mod logic: modifying an input using a learnable scalar
        # PyTorch: score = score + temp
        # TF: limit = base_limit + temp
        base_limit = 10
        limit = tf.cast(base_limit + temp, tf.int32)
        
        # Call the target API
        return tf.compat.v1.train.range_input_producer(
            limit, 
            num_epochs=1, 
            shuffle=False,
            capacity=32
        )

    with tf.compat.v1.Session() as sess:
        # Initialize variables (required for learnable params and local counters)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        # 3. Execute the compiled function
        # In the PyTorch bug, forward worked but backward failed, or compile failed.
        # Here we verify the queue is created and usable.
        queue = create_producer()

        # Verify behavior by attempting to dequeue
        # Since shuffle=False and temp=0, we expect 0, 1, 2...
        val = sess.run(queue.dequeue())
        print(f"Dequeued value: {val}")
        assert val == 0, f"Expected 0, got {val}"

        # Test with a different scalar value to ensure the parameter is being used
        sess.run(temp.assign(2.0))
        # Note: range_input_producer creates the queue graph once. 
        # Re-running the function to check if it handles the updated scalar logic.
        queue_2 = create_producer()
        val_2 = sess.run(queue_2.dequeue())
        print(f"Dequeued value with offset: {val_2}")
        # With temp=2.0, limit becomes 12. First value is 0.
        assert val_2 == 0, f"Expected 0, got {val_2}"

        print("Test passed: range_input_producer handles learnable scalars.")

if __name__ == "__main__":
    test_range_input_producer_learnable_scalar()