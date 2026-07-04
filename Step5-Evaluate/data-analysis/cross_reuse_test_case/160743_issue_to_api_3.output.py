import torch
import torch.nn as nn

def test_mps_avgpool2d_with_distributed_check():
    """
    Test case for Issue 160743: MPS AvgPool2d produces incorrect output.
    
    This test leverages the similar API 'torch.distributed.is_available' 
    to ensure the testing environment supports distributed checks, 
    mirroring the pattern of verifying backend capabilities before 
    executing specific operations.
    """
    
    # Leverage the similar API: Check distributed availability
    # This reflects the pattern of checking environment capabilities (like MPS availability)
    # before running the core logic, similar to how is_available() checks for _c10d_init.
    if not torch.distributed.is_available():
        print("Skipping test: torch.distributed is not available.")
        return

    # Check MPS availability specifically for the bug reproduction
    if not torch.backends.mps.is_available():
        print("Skipping test: MPS backend is not available.")
        return

    torch.manual_seed(0)

    # Original Bug Reproduction Logic
    # Model configuration that triggers the incorrect output on MPS
    model = nn.AvgPool2d(kernel_size=[1, 6], stride=[4, 9], ceil_mode=True, divisor_override=3)
    x = torch.randn(4, 6, 7)

    # Compute outputs
    out_cpu = model(x)
    out_mps = model(x.to("mps"))

    # Verify the fix: CPU and MPS outputs should match
    # The original bug showed a mismatch where MPS output had zeros.
    # We use assert to fail the test if the bug persists.
    assert torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2), (
        f"MPS AvgPool2d output mismatch detected.\n"
        f"CPU Output:\n{out_cpu}\n"
        f"MPS Output:\n{out_mps.cpu()}"
    )
    
    print("Test passed: CPU and MPS outputs match.")

if __name__ == "__main__":
    test_mps_avgpool2d_with_distributed_check()