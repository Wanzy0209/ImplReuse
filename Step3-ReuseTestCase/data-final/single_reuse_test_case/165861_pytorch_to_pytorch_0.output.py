import torch
import torch.nn.functional as F

def test_pad_reflect_large_batch():
    """
    Test case to verify the fix for Issue 165861:
    Reflect padding should not break when a batch dimension is larger than uint16 max (2**16).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    print("Testing torch.nn.functional.pad with large dimensions on CUDA...")

    # Case 1: Batch dimension > 2**16 (The reported bug case)
    # Expected: Should succeed (or fail if bug is present)
    try:
        x = torch.rand(2**16, 2, device="cuda")
        F.pad(x, (1, 1), mode="reflect")
        print(" Case 1 PASSED: Batch dim > 2**16 with reflect mode")
    except RuntimeError as e:
        print(f" Case 1 FAILED: Batch dim > 2**16 with reflect mode - {e}")

    # Case 2: Middle dimension > 2**16 (Another reported bug case)
    try:
        x = torch.rand(1, 2**16, 2, device="cuda")
        F.pad(x, (1, 1), mode="reflect")
        print(" Case 2 PASSED: Middle dim > 2**16 with reflect mode")
    except RuntimeError as e:
        print(f" Case 2 FAILED: Middle dim > 2**16 with reflect mode - {e}")

    # Case 3: Last dimension > 2**16 (Should work even with the bug)
    try:
        x = torch.rand(2, 2**16, device="cuda")
        F.pad(x, (1, 1), mode="reflect")
        print(" Case 3 PASSED: Last dim > 2**16 with reflect mode")
    except RuntimeError as e:
        print(f" Case 3 FAILED: Last dim > 2**16 with reflect mode - {e}")

    # Case 4: Batch dimension == 2**16 - 1 (Boundary check, should work)
    try:
        x = torch.rand(2**16 - 1, 2, device="cuda")
        F.pad(x, (1, 1), mode="reflect")
        print(" Case 4 PASSED: Batch dim == 2**16 - 1 with reflect mode")
    except RuntimeError as e:
        print(f" Case 4 FAILED: Batch dim == 2**16 - 1 with reflect mode - {e}")

    # Case 5: Other modes with large batch dimension (Should work)
    for mode in ["constant", "replicate", "circular"]:
        try:
            x = torch.rand(2**16, 2, device="cuda")
            F.pad(x, (1, 1), mode=mode)
            print(f" Case 5 PASSED: Batch dim > 2**16 with {mode} mode")
        except RuntimeError as e:
            print(f" Case 5 FAILED: Batch dim > 2**16 with {mode} mode - {e}")

if __name__ == "__main__":
    test_pad_reflect_large_batch()