import torch
import tensorflow as tf
import pytest

def test_range_input_producer_strict_mode():
    """
    Adapted test case for tf.compat.v1.train.range_input_producer.
    
    The original PyTorch issue (167927) describes a conflict between a strict mode 
    (fullgraph=True) and an exclusion mechanism (torch.compiler.disable). 
    The user expects the exclusion to be respected without error.
    
    For tf.compat.v1.train.range_input_producer, the 'strict' mode is defined by 
    setting `num_epochs`, which limits the number of times the range is iterated.
    This test verifies that the producer respects this strict limit and handles 
    the OutOfRange error gracefully, analogous to how the PyTorch API should 
    handle the graph break.
    """
    # Disable eager execution to use TF 1.x graph mode (required for this API)
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        limit = 5
        num_epochs = 2
        batch_size = 1

        # Setup the producer with a strict limit (num_epochs)
        # This is similar to torch.compile(fullgraph=True) where we expect specific behavior
        queue = tf.compat.v1.train.range_input_producer(
            limit=limit,
            num_epochs=num_epochs,
            shuffle=False,
            capacity=32
        )

        # Define the dequeue operation
        dequeue_op = queue.dequeue_many(batch_size)

        # Initialize local variables (required when num_epochs is not None)
        # This is analogous to setting up the compilation context
        init_local = tf.compat.v1.local_variables_initializer()
        init_global = tf.compat.v1.global_variables_initializer()
        sess.run([init_local, init_global])

        total_items = 0
        expected_items = limit * num_epochs

        try:
            while True:
                # Run the operation
                result = sess.run(dequeue_op)
                total_items += len(result)
        except tf.errors.OutOfRangeError:
            # In the PyTorch bug, an error is raised unexpectedly.
            # Here, OutOfRangeError is the expected termination condition for the strict mode.
            # We verify that it occurs at the correct time.
            pass

        # Assert that the producer respected the 'strict' num_epochs setting
        assert total_items == expected_items, \
            f"Expected {expected_items} items (limit * num_epochs), but got {total_items}"

if __name__ == "__main__":
    test_range_input_producer_strict_mode()