import torch
import tensorflow as tf
import os
import sys

def test_range_input_producer_command_injection():
    """
    Adapted test case for OS command injection based on PyTorch Issue #167480.
    
    Original Bug: torch.utils.cpp_extension.load_inline allowed shell injection 
                  via 'extra_cflags' because it used subprocess.check_output(shell=True).
    
    Target API: tf.compat.v1.train.range_input_producer.
    
    This test verifies that passing shell metacharacters to the TensorFlow API
    does not result in arbitrary command execution, as the API constructs a 
    graph queue rather than invoking a shell subprocess.
    """
    
    # Ensure we are using the v1 compat API behavior
    tf.compat.v1.disable_eager_execution()

    # Define a malicious payload intended to create a file if executed by a shell
    # In the PyTorch bug, this would be passed to extra_cflags.
    # Here, we attempt to pass it to string arguments like 'name' or 'shared_name'.
    malicious_payload = "; touch /tmp/tf_pwned_test"
    target_file = "/tmp/tf_pwned_test"

    # Clean up if file exists from previous runs
    if os.path.exists(target_file):
        os.remove(target_file)

    try:
        # Attempt to inject the payload via the 'name' parameter.
        # In the original PyTorch bug, the vulnerability triggered immediately 
        # upon calling the function because it internally called subprocess.
        # Here, we call the function to see if it triggers a shell execution.
        queue = tf.compat.v1.train.range_input_producer(
            limit=10, 
            name=malicious_payload
        )
        
        # Note: range_input_producer is a graph constructor. Even if we run the session,
        # the 'name' parameter is used as a scope name in the graph, not passed to a shell.
        # The vulnerability check is primarily that the function call itself doesn't 
        # trigger a shell command (unlike the PyTorch case).
        
    except Exception as e:
        # If the API raises an exception (e.g., invalid name format), that is acceptable 
        # and safe behavior. We only care if the command executes.
        print(f"API raised exception (safe behavior): {e}")

    # Verify the malicious command was NOT executed
    if os.path.exists(target_file):
        print("FAIL: OS command injection vulnerability detected! The file was created.")
        # Clean up the proof of concept file
        os.remove(target_file)
        sys.exit(1)
    else:
        print("PASS: No command injection detected. The API handled the input safely.")

if __name__ == "__main__":
    test_range_input_producer_command_injection()