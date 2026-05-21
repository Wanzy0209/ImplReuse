import torch
import collections
import tensorflow as tf

# The API under test: Enable eager execution
# This is the TensorFlow equivalent context to the PyTorch Dynamo issue,
# where we verify if the problematic construct (defaultdict) works
# when graph compilation is disabled (eager mode).
tf.compat.v1.enable_eager_execution()

def test_defaultdict_creation():
    """
    Adapted from PyTorch Dynamo Issue 166238.
    Original Bug: torch.compile (graph mode) fails to trace collections.defaultdict.
    This test verifies that collections.defaultdict works correctly
    when using tf.compat.v1.enable_eager_execution.
    """
    # Core logic from the bug report: creating a defaultdict
    # The original error was: "Unsupported function call ... <class 'collections.defaultdict'>"
    dd = collections.defaultdict(list)
    
    # Trigger the default factory behavior
    dd['a'].append(1)
    dd['b'].append(2)
    
    # Verify the behavior
    assert dd['a'] == [1], "defaultdict should handle key 'a' correctly"
    assert dd['b'] == [2], "defaultdict should handle key 'b' correctly"
    assert dd['c'] == [], "defaultdict should return empty list for missing key"
    
    # Verify the type
    assert isinstance(dd, collections.defaultdict)

if __name__ == "__main__":
    test_defaultdict_creation()
    print("Test passed successfully.")