import tensorflow as tf

def test_batch_parallel_duplicate_execution():
    """
    Test to verify that the computation passed to tf.compat.v1.tpu.batch_parallel
    is not executed multiple times due to a logic error (similar to the PyTorch 
    joint_custom_pre_pass merge mistake).
    """
    # Disable eager execution to ensure we are testing the graph compilation/execution behavior
    tf.compat.v1.disable_eager_execution()

    # Use a variable to track how many times the computation is executed
    # This acts as a side effect to detect duplicate runs.
    execution_counter = tf.compat.v1.get_variable(
        "execution_counter", 
        shape=[], 
        dtype=tf.int32, 
        initializer=tf.zeros_initializer()
    )

    def user_computation(inputs):
        """
        The user-defined computation. 
        If the API has a bug causing duplicate execution (like the PyTorch issue),
        this function's side effect (incrementing the counter) will trigger multiple times.
        """
        # Increment the counter every time this op is executed
        update_op = execution_counter.assign_add(1)
        
        with tf.control_dependencies([update_op]):
            # Perform a simple operation (identity or math)
            return inputs

    # Setup inputs: A list of lists of Tensors
    # Batch size 4, to be split into 2 shards
    inputs = [[tf.constant([1.0, 2.0, 3.0, 4.0])]]
    num_shards = 2

    # Call the API under test
    # This constructs the graph. If there is a merge mistake similar to PyTorch's,
    # the graph might contain duplicate nodes or the compilation might invoke the pass twice.
    outputs = tf.compat.v1.tpu.batch_parallel(
        user_computation, 
        inputs=inputs, 
        num_shards=num_shards
    )

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        try:
            # Run the computation
            sess.run(outputs)
            
            # Check the final value of the counter
            final_count = sess.run(execution_counter)
            
            # Expected behavior: The computation runs once per shard.
            # Bug behavior (PyTorch equivalent): The computation runs twice (duplicate invocation).
            # Note: Depending on implementation details, it might run num_shards times.
            # We check if it runs significantly more than expected (e.g., 2 * num_shards).
            expected_count = num_shards
            
            assert final_count == expected_count, (
                f"Likely merge mistake detected: Computation executed {final_count} times. "
                f"Expected {expected_count} times (once per shard)."
            )
            print("Test Passed: Computation executed the expected number of times.")
            
        except tf.errors.UnimplementedError as e:
            # Handle environments where TPU/XLA is not available
            print(f"Test skipped (TPU hardware unavailable): {e}")

if __name__ == "__main__":
    test_batch_parallel_duplicate_execution()