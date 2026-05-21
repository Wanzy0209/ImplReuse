import torch
from torch.amp import custom_bwd
from torch.autograd import Function
import os

# Test Case: OS Command Injection via torch.amp.custom_bwd
# Adapted from Issue 167480 (torch.utils.cpp_extension.load_inline)
# Original vulnerability: User input in 'extra_cflags' passed to shell=True.
# Similar API test: Passing shell metacharacters to 'device_type' in custom_bwd.

def test_custom_bwd_command_injection():
    # Payload attempting to create a file as proof of command execution
    # In the original bug, this would be passed to extra_cflags.
    # Here, we pass it to device_type.
    injection_payload = "cuda; touch /tmp/pwned_custom_bwd.txt #"
    
    # Cleanup if file exists from previous runs
    if os.path.exists("/tmp/pwned_custom_bwd.txt"):
        os.remove("/tmp/pwned_custom_bwd.txt")

    # Define a custom autograd function to utilize the decorator
    class TestFunction(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        @custom_bwd(device_type=injection_payload)
        def backward(ctx, grad_output):
            return grad_output

    try:
        # Execute forward and backward to trigger the logic
        x = torch.tensor([1.0], requires_grad=True)
        y = TestFunction.apply(x)
        y.sum().backward()

        # Check if the side effect (file creation) occurred
        if os.path.exists("/tmp/pwned_custom_bwd.txt"):
            print("FAIL: OS Command Injection vulnerability detected in torch.amp.custom_bwd!")
            os.remove("/tmp/pwned_custom_bwd.txt")
            return False
        else:
            print("PASS: No command execution detected (Input likely handled safely or ignored).")
            return True

    except ValueError as e:
        # Expected behavior: The API validates the device_type string and rejects invalid input
        print(f"PASS: ValueError raised as expected for invalid device_type: {e}")
        return True
    except Exception as e:
        # Catching other potential errors (e.g., RuntimeError if device type is checked during execution)
        print(f"INFO: Exception raised during execution: {type(e).__name__}: {e}")
        # As long as the command wasn't injected, this is a safe failure
        if not os.path.exists("/tmp/pwned_custom_bwd.txt"):
            print("PASS: No command execution detected despite exception.")
            return True
        else:
            print("FAIL: Command execution detected alongside exception.")
            os.remove("/tmp/pwned_custom_bwd.txt")
            return False

if __name__ == "__main__":
    test_custom_bwd_command_injection()