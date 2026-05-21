import torch
import tensorflow as tf
import numpy as np

def test_collapse_repeated_parameter_integrity():
    """
    Test case adapted from the torch.addmm bug report (Issue 167313).
    
    Original Bug: torch.compile replaced addmm with add(mm) but ignored 
    alpha/beta parameters, defaulting them to 1.
    
    Adaptation: This test checks if tf.nn.collapse_repeated respects the 
    'seq_length' parameter when run under tf.function (compiled mode), 
    ensuring parameters are not ignored or defaulted incorrectly.
    """
    
    # 1. Setup inputs
    # We create a sequence where the 'seq_length' parameter cuts off 
    # the end of the tensor. If 'seq_length' is ignored (like alpha/beta were),
    # the result will include data from the cut-off section.
    labels = tf.constant([[1, 1, 2, 2, 3, 3]], dtype=tf.int32)
    seq_length = tf.constant([4], dtype=tf.int32) # Only process first 4: [1, 1, 2, 2]

    # 2. Define the function using the similar API
    # Mirrors: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(...))
    f = lambda l, s: tf.nn.collapse_repeated(l, s)

    # 3. Compile the function
    # Mirrors: fc = torch.compile(f)
    fc = tf.function(f)

    # 4. Execute Eager and Compiled versions
    # Mirrors: f(x, a, b) vs fc(x, a, b)
    eager_labels, eager_len = f(labels, seq_length)
    compiled_labels, compiled_len = fc(labels, seq_length)

    # 5. Assertions
    
    # Check that compiled mode matches eager mode
    # (In the original bug, these differed significantly)
    assert np.array_equal(eager_labels.numpy(), compiled_labels.numpy()), \
        "Mismatch: Compiled mode ignored 'seq_length' parameter."
    assert np.array_equal(eager_len.numpy(), compiled_len.numpy()), \
        "Mismatch: Compiled mode produced incorrect sequence length."

    # Check specific correctness regarding the parameter
    # If seq_length=4 is respected, we process [1, 1, 2, 2] -> [1, 2].
    # The output should be [1, 2, 0, 0, 0, 0].
    # If seq_length is ignored (bug), we process [1, 1, 2, 2, 3, 3] -> [1, 2, 3].
    # The output would be [1, 2, 3, 0, 0, 0].
    expected_output = np.array([[1, 2, 0, 0, 0, 0]])
    
    assert np.array_equal(compiled_labels.numpy(), expected_output), \
        f"Expected {expected_output}, but got {compiled_labels.numpy()}. " \
        "This indicates the 'seq_length' parameter was ignored."

    print("Test passed: tf.nn.collapse_repeated respects seq_length in compiled mode.")

if __name__ == "__main__":
    test_collapse_repeated_parameter_integrity()