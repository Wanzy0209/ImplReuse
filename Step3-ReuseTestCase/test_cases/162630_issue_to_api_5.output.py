import torch
import torch.utils.dlpack
import torch.profiler.itt as itt

def test_dlpack_caching_with_profiler():
    """
    Test case for Issue 162630: Intrusive Caching DLPack for Fast Conversion.
    
    This test verifies the DLPack conversion logic while leveraging the similar API
    torch.profiler.itt.range_pop to instrument the conversion process. The issue
    highlights the overhead of metadata population during conversion, and this test
    uses the profiler to mark the scope of these operations.
    """
    # Create a sample tensor
    tensor = torch.randn(5, 5)

    # Instrument the first DLPack conversion
    # The similar API (range_pop) is used here to mark the end of the profiling range
    itt.range_push("DLPack_Conversion_1")
    dlpack_capsule_1 = torch.utils.dlpack.to_dlpack(tensor)
    itt.range_pop()

    # Instrument the second DLPack conversion
    # According to the RFC, this conversion should ideally utilize the cached
    # DLManagedTensorVersioned to reduce overhead.
    itt.range_push("DLPack_Conversion_2")
    dlpack_capsule_2 = torch.utils.dlpack.to_dlpack(tensor)
    itt.range_pop()

    # Verify that the conversions are valid by converting back to torch tensors
    # and checking equality.
    from_dlpack_1 = torch.utils.dlpack.from_dlpack(dlpack_capsule_1)
    from_dlpack_2 = torch.utils.dlpack.from_dlpack(dlpack_capsule_2)

    assert torch.equal(tensor, from_dlpack_1), "Data mismatch in first conversion"
    assert torch.equal(tensor, from_dlpack_2), "Data mismatch in second conversion"

    print("Test passed: DLPack conversions instrumented with ITT range_pop successfully.")

if __name__ == "__main__":
    test_dlpack_caching_with_profiler()