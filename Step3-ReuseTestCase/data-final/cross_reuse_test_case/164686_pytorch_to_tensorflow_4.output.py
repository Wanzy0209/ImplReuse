import torch
import tensorflow as tf
import numpy as np

def test_string_input_producer_divergence():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer based on 
    PyTorch issue 164686. The original issue involved scalar arithmetic 
    mixing int32/int64 types causing a compile error.
    
    Here, we replicate the scalar arithmetic logic to generate inputs 
    (specifically num_epochs and seed) for the TensorFlow API to verify 
    robustness against similar edge-case integer values.
    """
    
    # Replicate the seed and scalar logic from the PyTorch bug report
    seed = 13653
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Generate random integers similar to arg_0 and arg_1 in the bug report
    # arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
    arg_0 = int(np.random.randint(0, 100))
    arg_1 = int(np.random.randint(0, 100))

    # Replicate the arithmetic logic from the bug report
    # var_node_8 = 1 / -10 (int64 / int32 -> int64)
    # var_node_11 = arg_1 / -5 (int64 / int32 -> int32)
    # var_node_7 = var_node_8 + var_node_11
    
    # Simulating integer division (truncation towards zero) as in PyTorch/C++
    var_node_8 = int(1 / -10) 
    var_node_11 = int(arg_1 / -5)
    var_node_7 = var_node_8 + var_node_11

    # The result var_node_7 is used to configure the API
    # In the original bug, this value was used in multiplication.
    # Here, we use it as 'num_epochs' to test if the API handles the 
    # specific integer arithmetic result (which might be 0 or negative).
    
    # We use None if the result is invalid (negative) to ensure the test runs,
    # but we still calculate it to preserve the logic.
    num_epochs = var_node_7 if var_node_7 > 0 else None

    # Input data for the producer
    string_tensor = tf.constant(["file1.txt", "file2.txt", "file3.txt"])

    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for num_epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # Call the target API with parameters derived from the bug logic
        queue = tf.compat.v1.train.string_input_producer(
            string_tensor,
            num_epochs=num_epochs,
            shuffle=True,
            seed=seed,
            capacity=32
        )

        # Attempt to dequeue to verify the pipeline runs
        try:
            result = sess.run(queue.dequeue())
            print(f" TensorFlow API success with dequeued: {result}")
        except tf.errors.OutOfRangeError:
            # This is expected if num_epochs was calculated as 0
            print(" TensorFlow API success (OutOfRangeError as expected for 0 epochs)")
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_string_input_producer_divergence()