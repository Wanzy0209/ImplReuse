import torch
import tensorflow.compat.v1 as tf
import sys

# Disable TensorFlow v2 behavior to use compat.v1 APIs
tf.disable_v2_behavior()

def test_range_input_producer_regression():
    """
    Test case adapted from PyTorch Issue 161372 (torch.compile regression).
    
    Original Bug Description:
    The PyTorch issue involved a tensor size mismatch error when tensor sizes 
    changed from 77 to 78 during compilation, causing a recompile limit crash.
    
    Adaptation Logic:
    This test verifies that tf.compat.v1.train.range_input_producer correctly
    handles these specific limit values (77 and 78) to ensure no regression
    in range generation or boundary handling. We check if the producer correctly
    generates the sequence [0, limit-1] for both values mentioned in the bug report.
    """
    
    # Test limits extracted from the PyTorch bug report (expected 77, actual 78)
    limits_to_test = [77, 78]
    
    for limit in limits_to_test:
        print(f"Testing range_input_producer with limit={limit}...")
        
        # Reset the default graph to ensure a clean state for each iteration
        tf.reset_default_graph()
        
        # Create the range input producer
        # We use shuffle=False to ensure the sequence is deterministic (0, 1, 2, ...)
        producer = tf.compat.v1.train.range_input_producer(
            limit=limit,
            num_epochs=1,
            shuffle=False,
            capacity=limit + 10,
            name=f"range_producer_{limit}"
        )
        
        dequeue_op = producer.dequeue()
        
        with tf.Session() as sess:
            # Initialize local and global variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            # Start the queue runners to populate the queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            
            try:
                results = []
                # Attempt to dequeue exactly 'limit' items
                for _ in range(limit):
                    val = sess.run(dequeue_op)
                    results.append(val)
                
                # Assertion 1: Verify we received exactly 'limit' items
                assert len(results) == limit, \
                    f"Size mismatch: Expected {limit} items, but got {len(results)}"
                
                # Assertion 2: Verify the content matches the expected range [0, limit-1]
                expected_range = list(range(limit))
                assert results == expected_range, \
                    f"Content mismatch: Expected {expected_range}, but got {results}"
                
                print(f"  [PASS] Successfully verified range [0, {limit-1}]")
                
                # Assertion 3: Verify OutOfRangeError is raised when queue is empty
                # This mimics the strictness of the PyTorch size checks
                try:
                    sess.run(dequeue_op)
                    # If we reach here, the test fails because we expected an error
                    raise AssertionError(f"Expected OutOfRangeError for limit {limit}, but dequeue succeeded.")
                except tf.errors.OutOfRangeError:
                    print(f"  [PASS] Correctly raised OutOfRangeError after {limit} items")
                    
            except Exception as e:
                print(f"  [FAIL] Error encountered with limit {limit}: {e}")
                raise e
            finally:
                # Stop the queue runners
                coord.request_stop()
                coord.join(threads)

if __name__ == "__main__":
    try:
        test_range_input_producer_regression()
        print("\nAll tests passed successfully.")
    except Exception:
        print("\nTest failed.")
        sys.exit(1)