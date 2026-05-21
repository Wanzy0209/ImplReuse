import torch
import warnings
import sys

def test_no_futurewarning_on_import():
    """
    Test case to verify that importing the module containing the similar API
    (tensorflow.python.data.ops.iterator_ops) does not introduce FutureWarnings
    related to functools.partial in Enum definitions, similar to the bug
    reported in torch.distributed.algorithms.ddp_comm_hooks.
    """
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Import the module containing the similar API
        # The original bug was triggered by the import statement itself.
        try:
            import tensorflow.python.data.ops.iterator_ops
        except ImportError:
            print("TensorFlow is not installed. Skipping test.")
            return

        # Filter for the specific FutureWarning mentioned in the bug report
        # "functools.partial will be a method descriptor in future Python versions; 
        # wrap it in enum.member()..."
        partial_warnings = [
            warning for warning in w 
            if issubclass(warning.category, FutureWarning) 
            and "functools.partial" in str(warning.message)
            and "enum.member()" in str(warning.message)
        ]

        # Assert that no such warnings were raised
        assert len(partial_warnings) == 0, (
            f"Importing tensorflow.python.data.ops.iterator_ops introduced "
            f"{len(partial_warnings)} FutureWarning(s) related to functools.partial in Enum:\n"
            + "\n".join([f"- {str(w.message)}" for w in partial_warnings])
        )

if __name__ == "__main__":
    test_no_futurewarning_on_import()
    print("Test passed: No FutureWarnings detected.")