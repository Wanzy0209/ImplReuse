import torch
import tensorflow as tf
import numpy as np

def test_string_input_producer_global_state():
    """
    Adapted from PyTorch Issue 167064.
    
    The original bug report identified that calling torch.compile (redundantly)
    triggered a global side effect: 
    `torch.distributions.Distribution.set_default_validate_args(False)`.
    
    This test verifies that the TensorFlow equivalent API, 
    `tf.compat.v1.train.string_input_producer`, does not similarly 
    pollute global state (e.g., random seed or graph state) when called 
    but not executed.
    """
    
    # Ensure we are in a state where we can check global state.
    # We use the global random generator state as a proxy for global state integrity.
    # Note: string_input_producer is a compat.v1 API, often used in graph mode,
    # but we check eager-side state to see if the Python call itself leaks.
    
    # 1. Capture initial global random state
    # In TF 2.x, we check the global generator state.
    initial_state = tf.random.global_generator().state.numpy()
    
    # 2. Call the API (Redundant global code scenario)
    # We create a string tensor and call the producer.
    # We do NOT start the queue runners or run a session, mimicking the "never used" aspect.
    with tf.Graph().as_default():
        # Inside a graph context, string_input_producer adds ops.
        # We want to ensure this setup doesn't leak out to the global eager state.
        strings = tf.constant(["file1.txt", "file2.txt", "file3.txt"])
        
        # This call is analogous to the redundant _compiled_create_block_mask in PyTorch.
        # It sets up the pipeline but doesn't run it.
        _ = tf.compat.v1.train.string_input_producer(
            strings, 
            num_epochs=None, 
            shuffle=True, 
            seed=None, 
            capacity=32
        )
        
        # We do not call sess.run() or start_queue_runners().
        # The bug in PyTorch happened at definition/compile time, not execution time.

    # 3. Check final global random state
    final_state = tf.random.global_generator().state.numpy()

    # 4. Assert that the global state has not changed unexpectedly.
    # If the API call triggered a global side effect (like setting a seed or 
    # modifying a global config), this assertion would fail.
    np.testing.assert_array_equal(
        initial_state, 
        final_state,
        err_msg="Global random state was modified by string_input_producer setup. "
                "This indicates a side effect similar to the PyTorch bug."
    )

if __name__ == "__main__":
    test_string_input_producer_global_state()
    print("Test passed: No global state side effects detected.")