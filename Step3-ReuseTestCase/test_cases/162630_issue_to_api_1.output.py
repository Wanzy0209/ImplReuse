import torch
import torch.utils.dlpack
import torch.nn.functional as F
import threading

def test_dlpack_thread_safety_with_sigmoid():
    """
    Test case derived from Issue 162630 (Intrusive Caching DLPack).
    
    This test verifies the thread-safety and correctness of repeated DLPack 
    conversions (the scenario described in the issue) on tensors produced 
    by the similar API (torch.nn.functional.sigmoid).
    
    The original C++ implementation (fix) uses atomic operations to handle 
    concurrent access to the cached DLManagedTensor. This test simulates 
    that concurrent access to ensure the logic holds.
    """
    # Create a base tensor
    x = torch.randn(100, 100)
    
    # Leverage the similar API: torch.nn.functional.sigmoid
    # This generates the tensor that will be subject to frequent DLPack exchanges
    y = F.sigmoid(x)
    
    # Containers for thread results
    results = []
    errors = []

    def worker():
        try:
            # Simulate "frequent tensor exchanges" mentioned in the bug description
            # which trigger the caching logic in the C++ implementation.
            for _ in range(50):
                # Convert to DLPack (triggers ToDLPack in C++)
                capsule = torch.utils.dlpack.to_dlpack(y)
                
                # Convert back from DLPack
                z = torch.utils.dlpack.from_dlpack(capsule)
                
                # Store a clone to verify data integrity later
                results.append(z.clone())
        except Exception as e:
            errors.append(e)

    # Create multiple threads to stress-test the atomic compare-and-swap
    # logic present in the provided C++ code snippet.
    threads = []
    for _ in range(4):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Assert no errors occurred during concurrent access
    assert len(errors) == 0, f"Errors encountered during threading: {errors}"

    # Assert correctness: All converted tensors should match the original sigmoid tensor
    for res in results:
        assert torch.allclose(res, y), "Data mismatch detected after DLPack roundtrip"

if __name__ == "__main__":
    test_dlpack_thread_safety_with_sigmoid()
    print("Test passed.")