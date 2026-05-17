import tensorflow as tf

# Disable eager execution to use TF v1 graph mode and queues
tf.compat.v1.disable_eager_execution()

# Define input data
# In the original PyTorch bug, x.nonzero() produces a tensor with a data-dependent size.
# Here, we use a list of strings to feed the producer.
string_data = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]

# Use the similar API: tf.compat.v1.train.string_input_producer
# This creates a queue to hold the strings.
# num_epochs=1 ensures the queue stops after producing all strings once.
queue = tf.compat.v1.train.string_input_producer(string_data, shuffle=False, num_epochs=1)

# To mimic the tensor slicing (nz[:-1]) from the original bug, we need a tensor.
# string_input_producer outputs a scalar string from the queue. 
# We batch the output to create a 1D tensor, similar to how nonzero() returns a 2D tensor.
# allow_smaller_final_batch=True introduces dynamic shape behavior (last batch might be smaller).
batch_size = 3
batch_tensor = tf.compat.v1.train.batch([queue], batch_size=batch_size, capacity=32, allow_smaller_final_batch=True)

# Perform the slice operation (mimicking nz[:-1] from the original bug)
# This attempts to slice the tensor produced by the input pipeline.
sliced_tensor = batch_tensor[:-1]

# Initialize and run the session
with tf.compat.v1.Session() as sess:
    # Initialize local variables (required for epochs counter)
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    # Start queue runners to feed data into the queue
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # Run the operation to verify behavior
        # We loop to consume the batches from the queue
        print("Testing slicing of tensor from string_input_producer...")
        step = 0
        while not coord.should_stop():
            try:
                result = sess.run(sliced_tensor)
                print(f"Step {step} - Sliced Result: {result}")
                step += 1
            except tf.errors.OutOfRangeError:
                # Expected when num_epochs is reached
                print("Reached end of input queue (OutOfRangeError).")
                break
    except Exception as e:
        print(f"Error encountered during execution: {e}")
    finally:
        # Stop the queue runners
        coord.request_stop()
        coord.join(threads)

print("Test completed.")