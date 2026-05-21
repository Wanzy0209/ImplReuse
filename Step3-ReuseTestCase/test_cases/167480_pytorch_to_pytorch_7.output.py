import torch
import torch.distributed as dist
import os
import tempfile
import sys

def test_broadcast_object_list_command_injection():
    """
    Test case to verify that torch.distributed.broadcast_object_list
    does not suffer from OS command injection similar to the
    torch.utils.cpp_extension.load_inline vulnerability (Issue 167480).
    
    The original vulnerability allowed arbitrary command execution via 
    shell=True in subprocess calls when user input was passed to 
    compilation flags. This test verifies that passing similar payloads
    to broadcast_object_list does not result in command execution.
    """
    
    # Check if distributed is available
    if not dist.is_available():
        print("torch.distributed not available. Skipping test.")
        return

    # Setup a minimal single-process distributed environment
    # Using a temporary file for the init method to avoid port conflicts
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        try:
            dist.init_process_group(
                backend='gloo',
                init_method=f'file://{tmp_file.name}',
                rank=0,
                world_size=1
            )

            # Define a payload that would execute a command if passed to a shell
            # (e.g., via subprocess.check_output(..., shell=True))
            # Original bug context: extra_cflags="; touch /tmp/pwned"
            malicious_payload = "; touch /tmp/pwned"
            
            # Adapt the original call site:
            # Original: torch.utils.cpp_extension.load_inline(..., extra_cflags=malicious_payload)
            # Adapted: torch.distributed.broadcast_object_list(..., object_list=[malicious_payload])
            
            object_list = [malicious_payload]
            
            # Execute the similar API
            dist.broadcast_object_list(object_list, src=0)

            # Verification 1: Data Integrity
            # The object should be received as a string, matching the input
            assert object_list[0] == malicious_payload, "Data integrity check failed."
            
            # Verification 2: Security Check
            # The command should NOT have been executed.
            # If the file exists, it means the shell interpreted the payload.
            is_pwned = os.path.exists("/tmp/pwned")
            
            # Cleanup if vulnerability was triggered (though we expect it not to be)
            if is_pwned:
                os.remove("/tmp/pwned")

            assert not is_pwned, "Security vulnerability detected: Command injection succeeded!"

            print("Test Passed: No command injection vulnerability found in torch.distributed.broadcast_object_list.")

        except Exception as e:
            print(f"Test failed with exception: {e}")
            raise
        finally:
            # Ensure process group is destroyed
            if dist.is_initialized():
                dist.destroy_process_group()

if __name__ == "__main__":
    test_broadcast_object_list_command_injection()