import torch
import tensorflow as tf
import os
import sys

def test_gru_cell_command_injection():
    """
    Adapted test case for OS command injection vulnerability.
    
    Original PyTorch Issue: torch.utils.cpp_extension.load_inline allowed 
    OS command injection via the 'extra_cflags' argument when using 'use_pch=True',
    because it passed user input directly to subprocess.check_output(shell=True).
    
    This test attempts to verify if tf.keras.layers.GRUCell exhibits similar 
    behavior when handling string arguments (e.g., 'name'), although GRUCell 
    does not perform dynamic compilation.
    """
    
    # The malicious payload intended to execute a shell command.
    # In the PyTorch case, this was passed to extra_cflags.
    # Here, we attempt to pass it to the 'name' argument.
    malicious_payload = "; touch /tmp/pwned_by_tf"
    test_file_path = "/tmp/pwned_by_tf"

    # Clean up before test
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

    try:
        # Attempt to instantiate the layer with the malicious payload.
        # 'name' is a string argument that accepts user input.
        # If the underlying implementation blindly passes this to a shell,
        # the command would execute.
        cell = tf.keras.layers.GRUCell(units=10, name=malicious_payload)
        
        # Attempt to build the layer and call it to ensure any lazy initialization
        # or compilation (if any existed) is triggered.
        batch_size = 2
        inputs = tf.random.normal([batch_size, 10])
        initial_state = cell.get_initial_state(inputs=inputs)
        output, next_state = cell(inputs, initial_state)

    except Exception as e:
        print(f"Exception occurred during layer instantiation/execution: {e}")
        # We allow exceptions (e.g., invalid name format), but we check for the side effect.
    
    # Verify the vulnerability: Check if the file was created.
    if os.path.exists(test_file_path):
        print("FAIL: OS Command Injection vulnerability detected!")
        print(f"The file {test_file_path} was created.")
        os.remove(test_file_path)
        return False
    else:
        print("PASS: No OS command injection detected.")
        print("The API handled the input safely without executing shell commands.")
        return True

if __name__ == "__main__":
    # Run the test
    success = test_gru_cell_command_injection()
    sys.exit(0 if success else 1)