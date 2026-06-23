import torch

def test_quantile_mps_sliced():
    """
    Test case to verify torch.quantile behavior on MPS with sliced tensors.
    This is derived from a bug report where repeat_interleave crashed on MPS
    when using a sliced tensor (counts[1:3]).
    """
    if not torch.backends.mps.is_available():
        print("MPS is not available. Skipping test.")
        return

    # Setup: Create a tensor on MPS
    # Mimicking the pattern where a tensor is sliced before being passed to the function
    data = torch.arange(10, dtype=torch.float32, device="mps")
    
    # Create a non-prefix slice (similar to counts[1:3] in the bug report)
    # This tests if the MPS backend handles non-contiguous memory layouts correctly for quantile
    sliced_data = data[2:8]

    # Define quantile value
    q = 0.5

    try:
        # Call the similar API: torch.quantile
        result_mps = torch.quantile(sliced_data, q)

        # Verification: Compare with CPU result to ensure correctness
        data_cpu = data.cpu()
        sliced_data_cpu = data_cpu[2:8]
        result_cpu = torch.quantile(sliced_data_cpu, q)

        # Assert that the MPS result matches the CPU result
        assert torch.allclose(result_mps.cpu(), result_cpu), \
            f"MPS result {result_mps} does not match CPU result {result_cpu}"

        print("Test passed: torch.quantile works correctly with sliced tensors on MPS.")
        
    except RuntimeError as e:
        print(f"Test failed with RuntimeError: {e}")
    except Exception as e:
        print(f"Test failed with unexpected exception: {e}")

if __name__ == "__main__":
    test_quantile_mps_sliced()