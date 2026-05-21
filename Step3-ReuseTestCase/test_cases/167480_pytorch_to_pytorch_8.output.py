import torch
import torch.distributed as dist
import sys

def test_init_process_group_injection():
    """
    Test case to check for potential OS command injection in torch.distributed.init_process_group.
    This adapts the logic from the torch.utils.cpp_extension.load_inline vulnerability,
    where user-supplied strings were passed to a shell command via shell=True.
    """
    
    # We attempt to inject a shell command via the 'init_method' parameter.
    # If the underlying implementation uses subprocess with shell=True and does not sanitize
    # the input, this command could be executed.
    # Payload: "; echo 'COMMAND_INJECTION_TEST'"
    malicious_init_method = "tcp://127.0.0.1:29500; echo 'COMMAND_INJECTION_TEST'"

    try:
        # We use 'gloo' backend as it is commonly available and often involves process management.
        # rank=0, world_size=1 is used to attempt a single-process initialization for simplicity.
        dist.init_process_group(
            backend='gloo',
            init_method=malicious_init_method,
            rank=0,
            world_size=1
        )
        
        # If initialization succeeds without error, it might imply the input was sanitized
        # or the backend didn't invoke a shell. However, for this specific URL, it should fail
        # to connect, raising an exception.
        print("Initialization call returned successfully.")

    except RuntimeError as e:
        # We expect a RuntimeError (e.g., "Address already in use" or "Connection refused")
        # because the TCP socket is not listening.
        # If we see a shell error (like "command not found" printed directly to stderr
        # outside of the exception, or a specific OSError indicating shell failure),
        # that would indicate a vulnerability.
        print(f"Caught expected RuntimeError: {e}")
        
    except Exception as e:
        # Catching other exceptions to inspect potential shell-related errors.
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

    finally:
        # Clean up if the group was initialized
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_init_process_group_injection()