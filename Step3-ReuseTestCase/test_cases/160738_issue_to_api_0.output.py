import torch
import sys

def test_var_zero_dim_dim_zero():
    """
    Test case for Issue 160738: MPS torch.var cannot handle dim=0 on zero-dimensional input tensor.
    
    This test verifies that torch.var behaves consistently across CPU and MPS devices
    when called with dim=0 on a zero-dimensional tensor. The expected behavior is to
    return NaN (consistent with CPU), rather than raising a RuntimeError.
    """
    
    # 1. Test on CPU (Expected baseline behavior)
    x_cpu = torch.tensor(3.0)
    try:
        output_cpu = torch.var(x_cpu, dim=0)
        print(f"CPU Test: Success. Output: {output_cpu}")
        assert torch.isnan(output_cpu), "CPU output should be NaN"
    except Exception as e:
        print(f"CPU Test: Failed unexpectedly - {e}")
        sys.exit(1)

    # 2. Test on MPS (Target of the bug fix)
    if torch.backends.mps.is_available():
        x_mps = torch.tensor(3.0, device="mps")
        try:
            output_mps = torch.var(x_mps, dim=0)
            print(f"MPS Test: Success. Output: {output_mps}")
            
            # Verify consistency with CPU
            assert torch.isnan(output_mps), "MPS output should be NaN to match CPU behavior"
            assert output_mps.device.type == "mps", "Output device should be MPS"
            
        except RuntimeError as e:
            # This captures the specific error mentioned in the bug report:
            # "var_mps: reduction dim must be in the range of input shape"
            print(f"MPS Test: Failed with RuntimeError - {e}")
            print("This indicates the bug is present.")
            sys.exit(1)
        except Exception as e:
            print(f"MPS Test: Failed with unexpected error - {e}")
            sys.exit(1)
    else:
        print("MPS Test: Skipped (MPS device not available).")

if __name__ == "__main__":
    test_var_zero_dim_dim_zero()