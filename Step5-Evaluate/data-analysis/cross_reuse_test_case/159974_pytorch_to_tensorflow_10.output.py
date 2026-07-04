import tensorflow as tf

# Disable eager execution to use tf.compat.v1.train APIs
tf.compat.v1.disable_eager_execution()

# Define a function that wraps the API under test
# This mimics the structure of the original 'addcmul_func'
def range_producer_func(limit, num_epochs=None):
    return tf.compat.v1.train.range_input_producer(limit=limit, num_epochs=num_epochs, shuffle=False)

# Setup parameters (mimicking the tensor size 128 from the original bug report)
limit = 128
num_epochs = 1

# Create the queue
queue = range_producer_func(limit, num_epochs)

# Dequeue the elements to get the output tensor
# We dequeue all elements to verify the full range
output_tensor = queue.dequeue_many(limit)

# Run the graph
with tf.compat.v1.Session() as sess:
    # Initialize local variables (required for num_epochs counter)
    sess.run(tf.compat.v1.local_variables_initializer())
    
    # Start queue runners
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)

    try:
        # Run the operation
        result = sess.run(output_tensor)
        print("range_input_producer passed")
        
        # Basic assertion to verify behavior
        assert len(result) == limit, "Output length mismatch"
        assert list(result) == list(range(limit)), "Output values mismatch"
        
    finally:
        coord.request_stop()
        coord.join(threads)