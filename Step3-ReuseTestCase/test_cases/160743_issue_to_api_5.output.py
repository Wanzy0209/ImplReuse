import torch
import unittest

class TestMPSAvgPool2d(unittest.TestCase):
    """
    Test case to reproduce the MPS AvgPool2d incorrect output bug.
    Based on Issue ID: 160743.
    """
    
    def test_avgpool2d_mps_cpu_consistency(self):
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available on this system.")

        torch.manual_seed(0)

        # Initialize the model with parameters from the bug report
        # Note: divisor_override is a key parameter in this specific bug scenario
        model = torch.nn.AvgPool2d(
            kernel_size=[1, 6], 
            stride=[4, 9], 
            ceil_mode=True, 
            divisor_override=3
        )

        # Generate input data
        # Note: The bug report uses 3D input (4, 6, 7). 
        # Depending on the PyTorch version, nn.AvgPool2d might expect 4D input (N, C, H, W).
        # We preserve the original logic here. If running on a version requiring 4D, 
        # one might need to unsqueeze x to (4, 1, 6, 7).
        x = torch.randn(4, 6, 7)

        # Compute output on CPU
        out_cpu = model(x)

        # Compute output on MPS
        x_mps = x.to("mps")
        out_mps = model(x_mps)

        # Check if outputs match
        # The bug report indicates a mismatch where MPS output contains zeros where CPU does not.
        try:
            # We use a slightly relaxed tolerance as per the bug report (1e-2)
            match = torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2)
            
            if not match:
                print("Bug Reproduced: Output does not match!")
                print("CPU Output:\n", out_cpu)
                print("MPS Output:\n", out_mps.cpu())
                # Fail the test to indicate the bug presence
                self.fail("MPS and CPU outputs do not match.")
            else:
                print("Test Passed: Outputs match.")
                
        except Exception as e:
            print(f"Error during comparison: {e}")
            raise

if __name__ == "__main__":
    # Simple execution block to run the reproduction logic directly
    print("Running MPS AvgPool2d Reproduction Logic...")
    torch.manual_seed(0)
    
    model = torch.nn.AvgPool2d(kernel_size=[1, 6], stride=[4, 9], ceil_mode=True, divisor_override=3)
    x = torch.randn(4, 6, 7)
    
    out_cpu = model(x)
    
    if torch.backends.mps.is_available():
        out_mps = model(x.to("mps"))
        if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
            print("Output does not match!")
            print("CPU:\n", out_cpu)
            print("MPS:\n", out_mps.cpu())
        else:
            print("Outputs match.")
    else:
        print("MPS backend not available. Cannot run comparison.")