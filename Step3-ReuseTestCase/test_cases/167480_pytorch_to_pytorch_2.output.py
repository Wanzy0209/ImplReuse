import torch
import torch.library
import os

def test_register_vmap_command_injection():
    """
    Test case adapted from Issue 167480 (OS command injection).
    
    Original Vulnerability: torch.utils.cpp_extension.load_inline allowed OS command injection
    via extra_cflags when use_pch=True because it used subprocess.check_output(shell=True).
    
    Target API: torch.library.register_vmap
    
    This test verifies that passing shell metacharacters to torch.library.register_vmap
    does not result in OS command injection.
    """
    
    # Setup: Define a dummy custom op and vmap function
    def my_op_meta(x):
        return torch.empty_like(x)

    def my_vmap(info, in_dims, x):
        return x, 0

    # Payload: Shell metacharacters intended to execute a command.
    # In the original bug, passing this to extra_cflags with shell=True would execute 'touch /tmp/pwned'.
    malicious_op_name = "test_op; touch /tmp/pwned #"

    # Ensure the file doesn't exist before the test
    if os.path.exists("/tmp/pwned"):
        os.remove("/tmp/pwned")

    try:
        # Adaptation: Instead of load_inline(..., extra_cflags=...), we use register_vmap(..., op=...)
        # We pass the malicious string as the operator name.
        torch.library.define(f"{malicious_op_name}(Tensor x) -> Tensor", meta=my_op_meta)
        torch.library.register_vmap(malicious_op_name, my_vmap)
    except Exception:
        # If the API rejects the invalid name (e.g., ValueError), that is safe behavior.
        # We are only checking that arbitrary commands are NOT executed.
        pass

    # Verification: Check if the command was executed
    assert not os.path.exists("/tmp/pwned"), \
        "Security issue: OS command injection possible via torch.library.register_vmap"
    
    print("Test passed: torch.library.register_vmap is not vulnerable to OS command injection.")

if __name__ == "__main__":
    test_register_vmap_command_injection()