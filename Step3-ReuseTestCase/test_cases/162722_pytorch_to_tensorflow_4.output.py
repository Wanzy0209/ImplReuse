import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use TF v1 queue runners
tf.compat.v1.disable_eager_execution()

def test_string_input_producer_consistency():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    
    Original Bug Context: Severe numerical inconsistency with torch.compile() 
    (Eager mode vs Compiled mode).
    
    Adaptation Logic: Verify the consistency of the input pipeline across 
    multiple graph executions. Since string_input_producer is a data pipeline 
    API, we check if the output sequence is deterministic (consistent) when 
    a fixed seed is provided, analogous to checking if a compiled model 
    produces the same result as the eager model.
    """
    
    # 1. Setup Input Data
    # Simulating a dataset of file names
    input_strings = [f"file_{i}.bin" for i in range(10)]
    string_tensor = tf.constant(input_strings)

    # 2. Configure the API Under Test
    # We use shuffle=True with a fixed seed to test for consistency/determinism
    queue = tf.compat.v1.train.string_input_producer(
        string_tensor,
        num_epochs=2,      # Cycle through the data twice
        shuffle=True,      # Enable shuffling to test determinism
        seed=42,           # Fixed seed for reproducibility
        capacity=32
    )

    # Operation to dequeue data from the pipeline
    dequeue_op = queue.dequeue()

    results_run_1 = []
    results_run_2 = []

    # 3. Execution Run 1
    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for num_epochs counter)
        sess.run(tf.compat.v1.local_variables_initializer())
        
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                val = sess.run(dequeue_op)
                results_run_1.append(val.decode('utf-8'))
        except tf.errors.OutOfRangeError:
            # Expected when num_epochs is reached
            pass
        finally:
            coord.request_stop()
            coord.join(threads)

    # 4. Execution Run 2
    # We expect the exact same sequence of strings because the seed is fixed
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                val = sess.run(dequeue_op)
                results_run_2.append(val.decode('utf-8'))
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)

    # 5. Verification (Core Bug Reproduction Logic)
    # Check if the outputs are consistent (deterministic)
    assert len(results_run_1) > 0, "Run 1 produced no output"
    assert len(results_run_2) > 0, "Run 2 produced no output"
    
    # The core assertion: Consistency
    # If the API behaves inconsistently (like the bug report), this will fail.
    assert results_run_1 == results_run_2, (
        f"Severe Inconsistency detected between runs!\n"
        f"Run 1: {results_run_1}\n"
        f"Run 2: {results_run_2}"
    )
    
    # Verify correctness of epoch handling
    expected_length = len(input_strings) * 2
    assert len(results_run_1) == expected_length, (
        f"Expected {expected_length} items (2 epochs), got {len(results_run_1)}"
    )

    print("Test Passed: tf.compat.v1.train.string_input_producer behavior is consistent.")

if __name__ == "__main__":
    test_string_input_producer_consistency()