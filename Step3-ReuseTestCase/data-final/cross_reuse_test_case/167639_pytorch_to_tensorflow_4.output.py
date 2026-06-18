import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use v1 APIs and graph mode, 
# which is the native environment for string_input_producer
tf.compat.v1.disable_eager_execution()

def get_default_data():
    """Returns a list of strings to be used as input data."""
    return [f"file_{i}.txt" for i in range(10)]

def main():
    data = get_default_data()
    
    # Setup the string_input_producer
    # This mimics the 'model' definition in the PyTorch test.
    # We enable shuffle to trigger RNG usage, similar to the PyTorch bug context.
    queue = tf.compat.v1.train.string_input_producer(
        data, 
        shuffle=True, 
        seed=42, 
        capacity=32,
        num_epochs=1
    )
    
    # The output of string_input_producer is a string tensor (one dequeued item)
    output_tensor = queue

    # PyTorch Bug Logic: torch.compile inside torch.cuda.graph (Capture)
    # TensorFlow Adaptation: string_input_producer inside tf.xla.experimental.compile (Strict Compilation)
    # 
    # Note: string_input_producer relies on QueueRunners and stateful operations which are 
    # generally incompatible with XLA (just as RNG access was incompatible with CUDA Graph capture in PyTorch).
    # We attempt to run this to verify the behavior/incompatibility.
    
    print("Attempting to run string_input_producer inside XLA compile context (similar to CUDA Graph capture)...")
    try:
        # Attempt to compile the operation
        compiled_op = tf.xla.experimental.compile(output_tensor)
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord)
            
            # Try to run the compiled op
            result = sess.run(compiled_op)
            print("XLA Context Result:", result)
            
            coord.request_stop()
            coord.join(threads)
            
    except Exception as e:
        print(f"Error in XLA Context (Expected due to stateful queue ops): {e}")
        print("Falling back to standard Session execution to verify API functionality...")

        # Standard execution to verify the API works as intended outside the strict context
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord)
            
            # Run a few times to verify shuffling and output
            results = []
            for _ in range(5):
                val = sess.run(output_tensor)
                results.append(val)
                print(f"Standard Context Output: {val}")
            
            # Verify we got string outputs
            assert all(isinstance(r, bytes) for r in results), "Output should be bytes"
            print("API verification successful in standard context.")
            
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    main()