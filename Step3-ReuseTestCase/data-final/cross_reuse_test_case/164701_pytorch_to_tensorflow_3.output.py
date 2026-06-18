import tensorflow as tf

# Enable eager execution as requested by the API context
tf.compat.v1.enable_eager_execution()

# Check device availability to match the original "cuda" intent if possible
device_name = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"

print(f"Running test on device: {device_name}")

with tf.device(device_name):
    def slide_to_the_left_tf(state, new_events, arange, dev_null):
        batch_size = tf.shape(new_events)[0]

        # Concatenate state and new events
        concatenated = tf.concat([state, new_events], axis=1)

        # Replicate the advanced indexing logic from the PyTorch reproducer
        # PyTorch: batch_idx (B, 1), arange (1, N) -> concatenated[batch_idx, arange]
        # This acts as a complicated identity transformation
        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis]
        arange_expanded = arange[tf.newaxis, :]

        # Create coordinate grid for gather_nd
        # We need pairs (i, j) for all i in batch and j in arange
        grid_i = tf.tile(batch_idx, [1, tf.shape(arange)[0]])
        grid_j = tf.tile(arange_expanded, [batch_size, 1])
        
        # Flatten and stack to get indices of shape (B*N, 2)
        indices = tf.stack([
            tf.reshape(grid_i, [-1]), 
            tf.reshape(grid_j, [-1])
        ], axis=1)

        # Gather data and reshape back to (B, N, C)
        gathered = tf.gather_nd(concatenated, indices)
        concatenated = tf.reshape(gathered, [batch_size, tf.shape(arange)[0], tf.shape(concatenated)[2]])

        # Update state (In-place update equivalent)
        # state[:, :, :] = concatenated[:, -2048:, :]
        state.assign(concatenated[:, -2048:, :])

        # Update dev_null
        dev_null.assign(concatenated)

    # Initialize inputs
    # state must be a Variable to support in-place assignment
    state = tf.Variable(tf.zeros([4, 2048, 1024], dtype=tf.float32))
    
    # new_events: arange(1, 3) expanded to [4, 2, 1024]
    new_events = tf.broadcast_to(
        tf.range(1, 3, dtype=tf.float32)[tf.newaxis, :, tf.newaxis], 
        [4, 2, 1024]
    )
    
    arange = tf.range(2050, dtype=tf.int32)
    dev_null = tf.Variable(tf.zeros([4, 2050, 1024], dtype=tf.float32))

    # Execute the function
    slide_to_the_left_tf(state, new_events, arange, dev_null)

    # Verify the result
    # Only the last 2 rows should be non-zero (containing the new events)
    # The first 2046 rows should remain zero
    is_zero = tf.reduce_all(state[:, :-2, :] == 0)
    
    # Print result for debugging
    print(f"State check (first 2046 rows are zero): {is_zero.numpy()}")
    
    # Assert
    assert is_zero.numpy(), "Miscompilation/Logic error detected: State contains unexpected non-zero values."