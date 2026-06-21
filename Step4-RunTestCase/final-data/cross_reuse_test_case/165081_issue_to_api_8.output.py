import torch
import pytest

def test_nonzero_data_dependent_guard():
    """
    Test case for Issue 165081: [Fuzzer][Eager/Compile Divergence] 
    Could not guard on data-dependent expression Ne(u0, 9).
    
    This test verifies that torch.compile (torch._dynamo) can handle torch.nonzero
    operations where the output shape depends on the data (number of non-zero elements).
    
    It leverages the pattern from the similar API (tf.keras.initializers.serialize)
    by returning a tuple containing the result and a scalar metadata value (the count),
    mimicking the (byte_str, version) return structure.
    """
    
    # Check for torch._dynamo availability (PyTorch 2.0+)
    if not hasattr(torch, '_dynamo'):
        pytest.skip("torch._dynamo not available, skipping test (requires PyTorch 2.0+)")

    # Reproduce the configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    # Define the function to be compiled
    # Mimicking the return pattern of tf.keras.initializers.serialize (tuple of data and info)
    def find_nonzero_indices(x):
        indices = torch.nonzero(x)
        count = indices.size(0) # This is the data-dependent expression causing the guard issue
        return indices, count

    # Compile the function with dynamic shapes enabled to handle varying output sizes
    compiled_fn = torch.compile(find_nonzero_indices, dynamic=True)

    # Scenario 1: Input with exactly 9 non-zero elements
    # The error message "Ne(u0, 9)" suggests the compiler specialized on a size of 9.
    input_1 = torch.zeros(20, 20)
    input_1[0, :9] = 1.0 
    
    eager_indices_1, eager_count_1 = find_nonzero_indices(input_1)
    compiled_indices_1, compiled_count_1 = compiled_fn(input_1)

    assert torch.equal(eager_indices_1, compiled_indices_1), "Eager and compiled indices mismatch for size 9"
    assert eager_count_1 == compiled_count_1 == 9, "Count mismatch for size 9"

    # Scenario 2: Input with a different number of non-zero elements (e.g., 10)
    # This triggers the guard condition (Ne(u0, 9)) to test if the compiler handles the divergence.
    input_2 = torch.zeros(20, 20)
    input_2[0, :10] = 1.0 

    eager_indices_2, eager_count_2 = find_nonzero_indices(input_2)
    compiled_indices_2, compiled_count_2 = compiled_fn(input_2)

    assert torch.equal(eager_indices_2, compiled_indices_2), "Eager and compiled indices mismatch for size 10"
    assert eager_count_2 == compiled_count_2 == 10, "Count mismatch for size 10"

if __name__ == "__main__":
    try:
        test_nonzero_data_dependent_guard()
        print("Test passed successfully.")
    except pytest.exceptions.Skipped as e:
        print(f"Test skipped: {e}")