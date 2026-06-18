import torch

def test_isin_scalar_elements_compile():
    """
    Test case for torch.isin with scalar test_elements under torch.compile.
    This test leverages the batch processing pattern seen in tf.keras.backend.batch_set_value
    (iterating over a list of tuples) to verify the fix for the scalar input issue.
    """
    device = 'cpu' # Using CPU for general compatibility, issue applies to Inductor backend
    torch.manual_seed(777)

    # Define a list of test cases, mimicking the 'tuples' argument in batch_set_value.
    # Each tuple contains (input_elements, test_elements).
    # We specifically include a scalar test_elements (0-d tensor) to reproduce the bug.
    test_tuples = [
        (torch.randint(-50, 50, (5,), dtype=torch.int64, device=device), 
         torch.randint(-50, 50, (3,), dtype=torch.int64, device=device)), # Normal case
        (torch.randint(-50, 50, (5,), dtype=torch.int64, device=device), 
         torch.randint(-50, 50, (), dtype=torch.int64, device=device)),    # Scalar case (Bug trigger)
    ]

    # Helper function to be compiled
    def run_isin(elem, test_elem):
        return torch.isin(elem, test_elem, assume_unique=False, invert=False)

    # Compile the function using the Inductor backend
    compiled_fn = torch.compile(run_isin, backend='inductor')

    print("Running tests...")
    for i, (x, y) in enumerate(test_tuples):
        print(f"\nTest Case {i+1}:")
        print(f"  Input shape: {x.shape}, Test elements shape: {y.shape}")

        # Eager execution
        eager_out = run_isin(x, y)
        print(f"  Eager Result: {eager_out}")

        # Compiled execution
        try:
            compiled_out = compiled_fn(x, y)
            print(f"  Compiled Result: {compiled_out}")

            # Verify consistency
            assert torch.equal(eager_out, compiled_out), \
                f"Mismatch between eager and compiled results for case {i+1}"
            print("  Status: Passed")
        except Exception as e:
            print(f"  Status: Failed with error: {e}")
            raise

if __name__ == "__main__":
    test_isin_scalar_elements_compile()