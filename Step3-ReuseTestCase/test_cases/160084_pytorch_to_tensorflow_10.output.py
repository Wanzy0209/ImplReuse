import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1 APIs effectively
tf.compat.v1.disable_eager_execution()

def test_range_input_producer():
    """
    Adapted test case for tf.compat.v1.train.range_input_producer.
    
    Original Bug Context:
    The original PyTorch test case verified that torch.compile (backend="inductor") 
    could handle a model with parameters, conditional logic, and arithmetic operations 
    without crashing (RuntimeError: opt_ready_stream && opt_parent_stream).
    
    Adaptation Logic:
    This test adapts the scenario to TensorFlow by using tf.compat.v1.train.range_input_producer
    to generate inputs, applying similar arithmetic operations (x * a + b), and verifying
    the execution flow within a session, preserving the 'first_batch' check logic.
    """
    
    # 1. Setup parameters (mimicking RegressionModel __init__)
    a_val = 2.0
    b_val = 3.0
    limit = 10
    
    # 2. Define the input producer (The API under test)
    # This replaces the torch.randn inputs in the original test
    range_producer = tf.compat.v1.train.range_input_producer(
        limit=limit,
        num_epochs=1,
        shuffle=False,
        capacity=32,
        name="range_input_producer_test"
    )
    
    # 3. Define the model graph (mimicking RegressionModel forward)
    # Dequeue the integer input
    x = range_producer.dequeue()
    
    # Cast to float to match the original model's float parameters
    x_float = tf.cast(x, tf.float32)
    
    # Define parameters a and b
    a = tf.constant(a_val, dtype=tf.float32, name="param_a")
    b = tf.constant(b_val, dtype=tf.float32, name="param_b")
    
    # The operation: x * a + b
    y = x_float * a + b
    
    # 4. Execution (mimicking the compiled model execution)
    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for num_epochs) and global variables
        sess.run([tf.compat.v1.local_variables_initializer(), 
                  tf.compat.v1.global_variables_initializer()])
        
        # Start queue runners (analogous to starting the compiled backend)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)
        
        first_batch = True
        
        try:
            # Run a few steps to verify functionality
            for _ in range(5):
                # Execute the graph
                result = sess.run(y)
                
                if first_batch:
                    # Mimic the print statement from the original bug report
                    # Checking dtypes to ensure type consistency
                    print(f"Param a dtype: {a.dtype}, Param b dtype: {b.dtype}. Input dtype: {x_float.dtype}")
                    print(f"First batch result: {result}")
                    
                    # Assertions to verify correct behavior
                    assert result is not None, "Result should not be None"
                    assert isinstance(result, (np.floating, float)), "Result should be a float type"
                    
                    first_batch = False
        finally:
            # Stop the queue runners
            coord.request_stop()
            coord.join(threads)
            
    print("Test completed successfully.")

if __name__ == "__main__":
    test_range_input_producer()