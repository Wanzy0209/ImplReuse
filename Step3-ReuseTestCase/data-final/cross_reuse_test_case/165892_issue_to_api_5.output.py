import torch

def test_quantized_sigmoid_compile():
    """
    Test case based on Issue 165892 (torch.bmm + torch.compile failure with out_dtype).
    This test applies the same pattern to torch.ops.quantized.sigmoid, which takes
    output scale and zero_point arguments analogous to out_dtype.
    """
    # Setup: Quantized operations typically run on CPU
    input_tensor = torch.rand((1, 1024, 1024))
    input_quantized = torch.quantize_per_tensor(
        input_tensor, scale=1.0, zero_point=0, dtype=torch.quint8
    )

    # Output parameters (analogous to out_dtype in the original bug)
    output_scale = 1.0 / 256.0
    output_zero_point = 128

    # Reproduce the bug pattern: @torch.compile wrapping a function call
    # with specific output arguments.
    @torch.compile
    def run_quantized_sigmoid(input, scale, zero_point):
        return torch.ops.quantized.sigmoid(input, scale, zero_point)

    try:
        # Execute
        output = run_quantized_sigmoid(input_quantized, output_scale, output_zero_point)

        # Assertions to verify correctness
        assert output is not None, "Output should not be None"
        assert output.qscheme() == torch.per_tensor_affine, "Output should be quantized"
        assert output.shape == input_tensor.shape, "Output shape should match input"
        
        print("Test passed: torch.compile works with torch.ops.quantized.sigmoid output arguments.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_quantized_sigmoid_compile()