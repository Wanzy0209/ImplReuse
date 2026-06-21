import torch

def test_quantized_searchsorted():
    # Create a sorted sequence and values to search for
    # searchsorted requires the first argument to be sorted
    sorted_sequence_float = torch.tensor([1.0, 2.0, 3.0])
    values_float = torch.tensor([1.0, 2.5])

    # Quantize the inputs to QuantizedCPU
    scale = 1.0
    zero_point = 0
    dtype = torch.quint8

    sorted_sequence = torch.quantize_per_tensor(sorted_sequence_float, scale=scale, zero_point=zero_point, dtype=dtype)
    values = torch.quantize_per_tensor(values_float, scale=scale, zero_point=zero_point, dtype=dtype)

    print("Quantized sorted sequence:", sorted_sequence)
    print("Quantized values:", values)

    # Adapted call site: Attempt to use torch.searchsorted with QuantizedCPU tensors
    # This replaces the original torch.zeros_like call to test the similar API
    try:
        out_indices = torch.searchsorted(sorted_sequence, values)
        
        print("Output indices:", out_indices)
        
        # Verify correctness against the float implementation if it succeeds
        expected_indices = torch.searchsorted(sorted_sequence_float, values_float)
        assert torch.equal(out_indices, expected_indices), "Indices do not match float version"
        print("Test passed: torch.searchsorted works with QuantizedCPU tensors.")
        
    except NotImplementedError as e:
        print(f"NotImplementedError encountered: {e}")
        print("Test failed: torch.searchsorted not implemented for QuantizedCPU tensors.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_quantized_searchsorted()