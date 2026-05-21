import torch
import tensorflow as tf
import os
import sys

# Ensure we are using compat.v1 behavior as requested by the API name
tf.compat.v1.disable_eager_execution()

def test_tf_shuffle_batch_command_injection():
    """
    Adapts the PyTorch OS command injection test to tf.compat.v1.train.shuffle_batch.
    
    Original Bug Logic:
    The PyTorch API `torch.utils.cpp_extension.load_inline` with `use_pch=True` 
    constructed a shell command using user-supplied strings (e.g., `extra_cflags`) 
    and executed it via `subprocess.check_output(..., shell=True)`. 
    This allowed an attacker to inject arbitrary OS commands (e.g., '; touch /tmp/pwned').

    Adaptation:
    This test attempts to inject a similar payload into the TensorFlow API.
    Since `tf.compat.v1.train.shuffle_batch` is a graph operation and not a shell 
    command wrapper, we verify that passing malicious strings to its parameters 
    (specifically `name`, which is a string argument) does not result in OS command execution.
    """
    
    # The malicious payload intended to execute a shell command.
    # In the PyTorch bug, this was passed to `extra_cflags`.
    # Here we attempt to pass it to the `name` parameter.
    malicious_payload = "; touch /tmp/tf_pwned"
    
    # Setup a simple tensor pipeline
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor = tf.constant(input_data, dtype=tf.float32, name="input_tensor")

    print(f"Attempting to inject payload: '{malicious_payload}' into tf.compat.v1.train.shuffle_batch")

    try:
        # Attempt to inject the payload via the 'name' parameter
        batch = tf.compat.v1.train.shuffle_batch(
            [tensor],
            batch_size=2,
            capacity=10,
            min_after_dequeue=2,
            name=malicious_payload
        )

        with tf.compat.v1.Session() as sess:
            # Initialize variables and queue runners
            sess.run(tf.compat.v1.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Run the operation
            # We run it once to trigger the graph execution
            result = sess.run(batch)
            print(f"Operation executed. Result: {result}")

            # Stop the queue runners
            coord.request_stop()
            coord.join(threads)

    except Exception as e:
        print(f"Exception occurred during execution: {e}")

    # Verification: Check if the command was executed
    # If the API is vulnerable to OS command injection, this file will exist.
    if os.path.exists("/tmp/tf_pwned"):
        print("FAIL: OS Command Injection successful in TensorFlow API.")
        os.remove("/tmp/tf_pwned")
        return False
    else:
        print("PASS: TensorFlow API is not vulnerable to this OS Command Injection vector.")
        return True

if __name__ == "__main__":
    # Run the test
    success = test_tf_shuffle_batch_command_injection()
    sys.exit(0 if success else 1)