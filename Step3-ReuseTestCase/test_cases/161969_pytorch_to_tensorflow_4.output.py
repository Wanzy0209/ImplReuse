import torch
import tensorflow as tf

# Disable v2 behavior to use the v1 API as requested
tf.compat.v1.disable_v2_behavior()

def example_function():
    """
    Adapted from the PyTorch example to use tf.compat.v1.train.string_input_producer.
    The original logic involved linear algebra and compilation. 
    Here we adapt the structure to test the string input producer pipeline.
    """
    # Define the input data (strings instead of matrices)
    string_tensor = tf.constant(["file1.txt", "file2.txt", "file3.txt"])

    # Use the target API: string_input_producer
    # This creates a queue to output the strings
    # Note: shuffle=False to mimic deterministic behavior for testing
    queue = tf.compat.v1.train.string_input_producer(
        string_tensor, 
        num_epochs=1, 
        shuffle=False,
        capacity=32
    )

    # Dequeue the next string
    result = queue.dequeue()

    # Mimic the 'print' line from the PyTorch bug report to check for side effects
    # In TF v1 graph mode, tf.print adds a print op to the graph.
    # print_op = tf.print("Current string:", result) 

    return result

if __name__ == "__main__":
    # Adapted from the PyTorch execution block
    
    # In PyTorch: device = torch.device("mps")
    # In TF: We use the default session/graph context
    
    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for num_epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start the queue runners (equivalent to starting the data pipeline)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Run the function multiple times to verify behavior
            # PyTorch: res = compiled_function(data, p)
            # Here we dequeue items from the producer
            results = []
            for i in range(3):
                res = sess.run(example_function())
                results.append(res)
                print(f"Iteration {i}: {res}")

            # Verify consistency (checking if we got the expected strings in order)
            expected = [b"file1.txt", b"file2.txt", b"file3.txt"]
            assert results == expected, f"Expected {expected}, but got {results}"
            print("Test passed: String input producer behaved consistently.")

        except Exception as e:
            print(f"Error encountered: {e}")
        finally:
            # Stop the queue runners
            coord.request_stop()
            coord.join(threads)