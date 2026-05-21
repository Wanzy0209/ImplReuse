import torch
import tensorflow as tf
import numpy as np

# The API tf.compat.v1.train.string_input_producer is a legacy V1 API.
# It requires graph mode execution (disabling eager execution) to function correctly.
tf.compat.v1.disable_eager_execution()

def test_string_input_producer_correctness():
    """
    Adapts the PyTorch test_triton_kernel_to_cpu logic to TensorFlow.
    
    Semantic Mapping:
    - User-defined Triton kernel (add_kernel) -> string_input_producer (Data Producer)
    - GPU Tensor (out) -> Queue (Internal Buffer)
    - .cpu() (Device Transfer) -> queue.dequeue() (Queue Transfer)
    - out_cpu + 1 (CPU Op) -> tf.strings.join (String Op)
    """
    
    # Input data (equivalent to x, y tensors in PyTorch)
    input_strings = [f"file_{i}" for i in range(4)]
    
    # Graph construction (equivalent to the scope of torch.compile)
    with tf.compat.v1.Session() as sess:
        # 1. The "Kernel": Producer
        # Mimics the custom kernel writing data into a buffer (the queue)
        queue = tf.compat.v1.train.string_input_producer(
            input_strings, 
            num_epochs=1, 
            shuffle=False, 
            capacity=32
        )
        
        # 2. The "Transfer": Dequeue
        # Mimics out.cpu() - moving data from the asynchronous buffer to the current context
        filename = queue.dequeue()
        
        # 3. The "Operation": CPU processing
        # Mimics out_cpu + 1
        processed_filename = tf.strings.join([filename, "_processed"])
        
        # Initialization
        # Required for num_epochs counter (local variables)
        local_init = tf.compat.v1.local_variables_initializer()
        global_init = tf.compat.v1.global_variables_initializer()
        sess.run([global_init, local_init])
        
        # Start Queue Runners
        # Mimics the asynchronous execution of the GPU kernel
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)
        
        # Verification
        expected_results = [s + "_processed" for s in input_strings]
        actual_results = []
        
        try:
            # Run the graph to get results
            for _ in range(len(input_strings)):
                result = sess.run(processed_filename)
                actual_results.append(result.decode('utf-8'))
        except tf.errors.OutOfRangeError:
            # Expected when num_epochs is reached
            pass
        finally:
            coord.request_stop()
            coord.join(threads)
        
        # Assert correctness (equivalent to self.assertEqual)
        assert actual_results == expected_results, \
            f"Correctness issue: Expected {expected_results}, but got {actual_results}"
        
        print("Test passed: Data transfer and processing correct.")

if __name__ == "__main__":
    test_string_input_producer_correctness()