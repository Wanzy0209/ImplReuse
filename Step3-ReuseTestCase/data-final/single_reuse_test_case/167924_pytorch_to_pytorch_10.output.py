import torch

# Test case adapted for torch.ones based on the bug report involving torch.arange
# Bug: Crash on MPS when using repeat_interleave with sliced tensor

if torch.backends.mps.is_available():
    counts = torch.tensor([0, 1, 0], device="mps")
    # Adaptation: Replace torch.arange with torch.ones
    data = torch.ones(2, device="mps")

    # The operation that caused the crash in the original report
    # counts[1:3] results in a sliced tensor [1, 0]
    result = data.repeat_interleave(counts[1:3], dim=0)

    # Verify correctness
    # data is [1, 1], counts are [1, 0]. Result should be [1].
    expected = torch.ones(1, device="mps")
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"
else:
    print("MPS device not available. Test skipped.")