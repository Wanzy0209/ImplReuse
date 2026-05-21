import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1 graph-based APIs
tf.compat.v1.disable_eager_execution()

def test_string_input_producer_behavior():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    
    Original PyTorch Logic:
    1. Input a tensor [1, 2].
    2. Use tolist() to unpack elements into Python scalars (u0, u1).
    3. Perform operations using these scalars.
    
    Adapted TensorFlow Logic:
    1. Input a list of strings (filenames).
    2. Use string_input_producer to enqueue strings.
    3. Dequeue elements to retrieve them (analogous to unpacking).
    4. Verify the retrieval and usage within the session.
    """
    
    # Input data: analogous to torch.tensor([1, 2])
    input_strings = ["file0.txt", "file1.txt"]
    
    with tf.compat.v1.Session() as sess:
        # Create the string input producer
        # This sets up the queue and enqueues the strings
        queue = tf.compat.v1.train.string_input_producer(
            input_strings, 
            num_epochs=1, 
            shuffle=False
        )
        
        # Dequeue operation: analogous to u0, u1 = a.tolist()
        # In TF v1, we dequeue to get the next item from the queue
        dequeue_op = queue.dequeue()
        
        # Initialize local variables (required for num_epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Start the queue runners to populate the queue
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        
        results = []
        try:
            # Retrieve elements: analogous to iterating through the list returned by tolist()
            for _ in range(len(input_strings)):
                val = sess.run(dequeue_op)
                results.append(val)
                # In the original PyTorch code: return a*u0*u1
                # Here we simply verify we can extract and use the value
                print(f"Retrieved: {val}")
                
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)
            
    # Assertions to verify behavior
    assert len(results) == 2, "Expected to retrieve 2 items"
    # TF strings are returned as bytes
    assert results[0] == b"file0.txt", f"Expected file0.txt, got {results[0]}"
    assert results[1] == b"file1.txt", f"Expected file1.txt, got {results[1]}"
    
    print("Test passed: string_input_producer successfully enqueued and dequeued items.")

if __name__ == "__main__":
    test_string_input_producer_behavior()