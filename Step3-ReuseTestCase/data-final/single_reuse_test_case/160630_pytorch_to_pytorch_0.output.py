import torch

def test_zeros_like_quantized():
    """
    Test case to verify that torch.zeros_like works correctly with QuantizedCPU tensors.
    This addresses the bug where 'aten::zero_' was not implemented for the QuantizedCPU backend.
    """
    # Create a quantized tensor
    quant_input = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0, 3.0]), 
        scale=1.0, 
        zero_point=0, 
        dtype=torch.quint8
    )
    
    # Call torch.zeros_like (the API under test)
    # This previously raised: NotImplementedError: Could not run 'aten::zero_' with arguments from the 'QuantizedCPU' backend
    out_tensor = torch.zeros_like(quant_input)
    
    # Verify the output tensor properties
    assert out_tensor.shape == quant_input.shape, f"Shape mismatch: expected {quant_input.shape}, got {out_tensor.shape}"
    assert out_tensor.dtype == quant_input.dtype, f"Dtype mismatch: expected {quant_input.dtype}, got {out_tensor.dtype}"
    assert out_tensor.device == quant_input.device, f"Device mismatch: expected {quant_input.device}, got {out_tensor.device}"
    
    # Verify the content. 
    # With scale=1.0 and zero_point=0, the quantized representation of 0.0 is 0.
    # We check the integer representation to ensure the tensor is actually zeroed.
    assert torch.all(out_tensor.int_repr() == 0), "Output tensor values are not zero"
    
    print("Test passed: torch.zeros_like works with QuantizedCPU tensors.")

if __name__ == "__main__":
    test_zeros_like_quantized()