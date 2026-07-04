import torch
import pytest

def is_pytorch_built_with_rocm():
    """
    Translates the semantic of tf.test.is_built_with_rocm to PyTorch.
    This leverages the similar API pattern to check the build environment,
    which is relevant as the bug report details specific CUDA/ROCm versions.
    """
    return torch.version.hip is not None

def test_mvlgamma_floating_point_exception():
    """
    Test case for Issue 161871: Floating point exception in torch.Tensor.mvlgamma_
    
    This test reproduces the logic that caused a Floating Point Exception (FPE).
    The original bug involved passing an int64 tensor and a large 'p' value (1024).
    """
    # Log environment information, reflecting the similar API's focus on build details
    print(f"PyTorch version: {torch.__version__}")
    print(f"Built with ROCm: {is_pytorch_built_with_rocm()}")

    # Reproduce the original bug logic
    # Create an int64 tensor as in the original report
    tensor = torch.randint(low=0, high=10, size=(5,), dtype=torch.int64)
    
    # Setup arguments mimicking the original unpacking pattern
    input_args = [tensor, 1024]
    input_kwargs = {}

    # The bug causes a hard crash (Floating Point Exception).
    # If the bug is fixed, this might raise a ValueError (for invalid input type/size)
    # or execute successfully.
    # We use a try/except block to catch standard exceptions if the fix converts the crash to a managed error.
    # Note: A true SIGFPE cannot be caught by Python's try/except, but this structure
    # validates the expected behavior if the crash is mitigated.
    try:
        torch.Tensor.mvlgamma_(*input_args, **input_kwargs)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        # If the library correctly handles the invalid input instead of crashing,
        # the test passes by catching the exception.
        # The original test logic incorrectly called pytest.fail here. Raising a standard
        # exception (like RuntimeError) is the expected graceful behavior compared to a hard crash.
        print(f"Caught expected exception (graceful handling instead of crash): {e}")
        # Do not fail the test; catching the exception means the FPE was mitigated.

if __name__ == "__main__":
    test_mvlgamma_floating_point_exception()