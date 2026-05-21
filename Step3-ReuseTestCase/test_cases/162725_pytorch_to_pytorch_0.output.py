import torch
import torch.nn.functional as F

def test_conv_transpose3d_compilation_consistency():
    """
    Test case to verify the consistency of torch.nn.functional.conv_transpose3d
    between eager execution and compiled (inductor) execution.
    Based on Issue ID: 162725.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    try:
        # Import internal test utilities to reproduce the exact failing inputs
        from torch.testing._internal.common_methods_invocations import op_db
    except ImportError:
        print("Skipping test: torch.testing._internal.common_methods_invocations not found.")
        return

    # Retrieve the operator database entry for conv_transpose3d
    # Index 192 corresponds to nn.functional.conv_transpose3d in the reported version
    try:
        op_dict = op_db[192]
    except IndexError:
        print("Skipping test: op_db index 192 not found (PyTorch version mismatch?).")
        return

    # Generate sample inputs for CUDA and float32
    inputs = list(op_dict.sample_inputs("cuda", torch.float32, requires_grad=False))

    # Define the function to be tested
    # Note: The original bug report uses a function that only accepts input and weight,
    # relying on default values for stride, padding, etc.
    def fn(x, w):
        return F.conv_transpose3d(input=x, weight=w)

    print(f"Testing {len(inputs)} sample inputs for conv_transpose3d...")

    for i, sample in enumerate(inputs):
        # Replicate the argument unpacking from the bug report
        eager_args = sample.input, *sample.args
        
        # Unpack input and weight. 
        # The bug report unpacks (x, w, b), implying sample.args contains weight and bias.
        # We only need x and w for the defined function.
        x = eager_args[0]
        w = eager_args[1]

        # Compile the function using inductor backend and max-autotune mode
        compiled = torch.compile(fn, backend="inductor", mode="max-autotune")

        # Execute eager and compiled versions
        res_eager = fn(x, w)
        res_compiled = compiled(x, w)

        # Verify consistency
        try:
            torch.testing.assert_close(res_eager, res_compiled)
        except AssertionError as e:
            print(f"\nMismatch found on sample {i}:")
            print(f"Input shape: {x.shape}, Weight shape: {w.shape}")
            print(f"Error: {e}")
            raise

    print("All tests passed.")

if __name__ == "__main__":
    test_conv_transpose3d_compilation_consistency()