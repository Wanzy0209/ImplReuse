import torch
import tensorflow as tf
import sys

def test_string_input_producer_dynamic_inputs():
    """
    Adapted from torch.compile regression (Issue 161372).
    
    The original PyTorch issue involved a regression where torch.compile failed
    due to dynamic tensor shapes (sequence length changing from 77 to 78),
    triggering a recompilation limit error.
    
    This test verifies the similar TensorFlow API (string_input_producer)
    handling of inputs with varying characteristics (string lengths), 
    ensuring the input pipeline remains stable with dynamic data.
    """
    # Disable v2 behavior to use compat.v1 APIs
    tf.compat.v1.disable_v2_behavior()

    # Simulate dynamic input sizes (analogous to the 77 vs 78 sequence length in the bug)
    # We use strings of varying lengths to stress the input pipeline.
    dynamic_strings = [
        "a", 
        "ab", 
        "abc", 
        "abcd", 
        "abcde"
    ]
    
    with tf.compat.v1.Session() as sess:
        # Create the input producer
        # num_epochs=None allows cycling, but we set it to 1 for a finite test
        string_queue = tf.compat.v1.train.string_input_producer(
            dynamic_strings, 
            num_epochs=1, 
            shuffle=False, 
            capacity=32
        )
        
        # Dequeue the next string
        dequeue_op = string_queue.dequeue()
        
        # Initialize local variables (required for num_epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        
        # Start the queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        
        collected_strings = []
        
        try:
            # Run the pipeline
            # In the PyTorch bug, the loop caused recompilation until the limit was hit.
            # Here we verify the TF pipeline handles the loop without error.
            while not coord.should_stop():
                val = sess.run(dequeue_op)
                collected_strings.append(val.decode('utf-8'))
                
                # Safety break to prevent infinite loop if logic fails
                if len(collected_strings) > len(dynamic_strings) + 5:
                    break
                    
        except tf.errors.OutOfRangeError:
            # Expected when num_epochs is reached
            pass
        finally:
            coord.request_stop()
            coord.join(threads)
            
        # Assertions to verify correct behavior
        assert len(collected_strings) == len(dynamic_strings), \
            f"Expected {len(dynamic_strings)} strings, got {len(collected_strings)}"
        
        assert collected_strings == dynamic_strings, \
            f"Data mismatch: expected {dynamic_strings}, got {collected_strings}"
            
        print("Test passed: string_input_producer handled dynamic inputs correctly.")

if __name__ == "__main__":
    test_string_input_producer_dynamic_inputs()