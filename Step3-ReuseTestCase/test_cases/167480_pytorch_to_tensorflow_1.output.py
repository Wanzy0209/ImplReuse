import torch
import tensorflow as tf
import os
import sys

def test_tf_warm_start_command_injection():
    """
    Adapted test case for OS command injection vulnerability.
    
    Original Bug: torch.utils.cpp_extension.load_inline allowed OS command injection 
    via user-supplied arguments (e.g., extra_cflags) passed to a shell execution.
    
    Target API: tf.compat.v1.train.warm_start
    Similarity: Cross-library similar API (handling user-supplied paths/strings).
    
    This test verifies if passing shell metacharacters into the checkpoint path
    results in command execution, similar to the PyTorch vulnerability.
    """
    
    # Define a marker file that would be created if command injection is successful
    marker_file = "/tmp/tf_injection_test_marker"
    
    # Clean up before test
    if os.path.exists(marker_file):
        os.remove(marker_file)

    # Construct a malicious checkpoint path attempting command injection.
    # Payload: "; touch /tmp/tf_injection_test_marker; #"
    # If the API blindly passes this to a shell (like the PyTorch bug), 
    # the touch command will execute.
    malicious_ckpt_path = f"non_existent_checkpoint; touch {marker_file}; #"

    print(f"Testing command injection with payload: {malicious_ckpt_path}")

    try:
        # Setup a minimal TF1 graph context required for warm_start
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            # Initialize variables to allow warm_start to run logic
            var = tf.compat.v1.get_variable("test_var", shape=[1], initializer=tf.zeros_initializer())
            sess.run(tf.compat.v1.variables_initializer([var]))

            # Attempt to call the API with the malicious input
            tf.compat.v1.train.warm_start(
                ckpt_to_initialize_from=malicious_ckpt_path,
                vars_to_warm_start="test_var"
            )
            
    except Exception as e:
        # We expect an exception (e.g., NotFoundError) because the checkpoint path is invalid.
        # The critical check is whether the side effect (file creation) occurred.
        print(f"Caught expected exception during API call: {type(e).__name__}: {e}")

    # Verification: Check if the marker file was created.
    if os.path.exists(marker_file):
        print("FAIL: OS Command Injection vulnerability detected! Marker file was created.")
        os.remove(marker_file)
        return False
    else:
        print("PASS: No OS command injection detected. API handled input safely.")
        return True

if __name__ == "__main__":
    # Run the test
    success = test_tf_warm_start_command_injection()
    sys.exit(0 if success else 1)