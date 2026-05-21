import torch
import warnings
import sys

def test_threading_options_no_future_warnings():
    """
    Test case to verify that importing and instantiating tf.data.ThreadingOptions
    does not introduce FutureWarnings related to functools.partial, similar to 
    the issue reported in torch.distributed.algorithms.ddp_comm_hooks (Issue 163938).
    
    The original bug was triggered by Python 3.13's behavior change regarding 
    functools.partial inside Enums. This test checks if the similar API pattern 
    in TensorFlow (defining class attributes via factory functions) exhibits 
    similar warnings.
    """
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        # Ensure all warnings are always triggered
        warnings.simplefilter("always")
        
        try:
            # Import the specific API
            from tensorflow.data import ThreadingOptions
            
            # Instantiate the class to trigger any potential descriptor behavior
            # that might be hidden on simple import
            options = ThreadingOptions()
            
        except ImportError:
            print("TensorFlow is not installed. Skipping test.")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

        # Filter for FutureWarnings
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
        
        # Specifically check for the functools.partial warning mentioned in the bug report
        partial_warnings = [
            warning for warning in future_warnings 
            if "functools.partial" in str(warning.message)
        ]

        # Assert that no such warnings were raised
        assert len(partial_warnings) == 0, (
            f"FutureWarnings related to functools.partial detected:\n"
            f"{[str(warning.message) for warning in partial_warnings]}"
        )

if __name__ == "__main__":
    test_threading_options_no_future_warnings()
    print("Test passed: No FutureWarnings detected for tf.data.ThreadingOptions.")