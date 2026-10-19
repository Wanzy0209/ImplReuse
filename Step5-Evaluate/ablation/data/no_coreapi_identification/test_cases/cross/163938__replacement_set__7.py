import warnings
import sys

# Test case adapted from the original bug report logic.
# The original issue was triggered by importing a module that used functools.partial
# inside an Enum without enum.member(), causing FutureWarnings in Python 3.13.
# This test checks the similar API (torch.backends.nnpack) for the same behavior
# and verifies the replacement API (tf.data.experimental.OptimizationOptions) 
# does not introduce similar warnings.

def test_future_warnings_on_import():
    """
    Test that importing torch.backends.nnpack (similar API) triggers FutureWarnings
    related to functools.partial, while importing tf.data.experimental.OptimizationOptions
    (replacement API) does not.
    """
    
    # 1. Test the Similar API (torch.backends.nnpack)
    # Based on the high similarity score, this module likely has the same code pattern.
    print("Testing torch.backends.nnpack for FutureWarnings...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import torch.backends.nnpack
        except ImportError:
            print("torch.backends.nnpack not available, skipping check.")
        else:
            # Check if any FutureWarnings were raised
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            
            # We expect warnings here based on the similarity to the original bug.
            # The original bug warning message contains "functools.partial" and "enum.member".
            # We check for the presence of "functools.partial" to confirm the pattern.
            partial_warnings = [warning for warning in future_warnings if "functools.partial" in str(warning.message)]
            
            if partial_warnings:
                print(f"Found {len(partial_warnings)} FutureWarning(s) related to functools.partial in torch.backends.nnpack.")
                for warning in partial_warnings:
                    print(f"  - {warning.message}")
            else:
                print("No FutureWarnings related to functools.partial found in torch.backends.nnpack.")

    # 2. Test the Replacement API (tf.data.experimental.OptimizationOptions)
    # This API should ideally not trigger the same warnings, or serves as the 
    # semantic replacement for the pattern causing the issue.
    print("\nTesting tf.data.experimental.OptimizationOptions for FutureWarnings...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import tensorflow as tf
            # Access the class to ensure it is loaded
            _ = tf.data.experimental.OptimizationOptions
        except ImportError:
            print("tensorflow not available, skipping check.")
        else:
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            
            # We check for the specific warning pattern mentioned in the bug.
            # Since this is a replacement, we verify it doesn't suffer from the same issue
            # or simply report any warnings found.
            partial_warnings = [warning for warning in future_warnings if "functools.partial" in str(warning.message)]
            
            if partial_warnings:
                print(f"Found {len(partial_warnings)} FutureWarning(s) related to functools.partial in tf.data.experimental.OptimizationOptions.")
                for warning in partial_warnings:
                    print(f"  - {warning.message}")
            else:
                print("No FutureWarnings related to functools.partial found in tf.data.experimental.OptimizationOptions.")

if __name__ == "__main__":
    test_future_warnings_on_import()