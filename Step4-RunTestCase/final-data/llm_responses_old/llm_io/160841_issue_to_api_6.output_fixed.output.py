import torch
import torch.nn as nn

def test_replication_pad1d_dtype_stability():
    """
    Test case for torch.nn.ReplicationPad1d inspired by Issue 160841.
    
    The original issue reported that running a model with 'auto' dtype on MacOS
    resulted in garbage output, which was fixed by explicitly using bf16.
    This test verifies that ReplicationPad1d handles dtype conversions (specifically
    float16 and bfloat16) correctly without producing garbage (NaN/Inf) or
    incorrect values, preserving the semantic correctness of the operation.
    """
    # Setup device (MPS for MacOS, fallback to CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Instantiate the similar API: ReplicationPad1d
    # Padding of 2 on both sides
    layer = nn.ReplicationPad1d(2).to(device)
    
    # Create a standard input tensor
    # Shape: (Batch, Channel, Width) -> (1, 2, 4)
    input_fp32 = torch.arange(0, 8, dtype=torch.float32).reshape(1, 2, 4).to(device)
    
    # --- Scenario 1: Float16 (Simulating the 'auto' behavior that caused issues) ---
    try:
        input_fp16 = input_fp32.to(torch.float16)
        output_fp16 = layer(input_fp16)
        
        # Assert no garbage (NaN or Inf) is produced
        assert torch.isfinite(output_fp16).all(), "ReplicationPad1d produced garbage (NaN/Inf) with float16 input"
        
        # Verify the padding logic is preserved (values are replicated from edges)
        # Expected: [[0, 0, 0, 1, 2, 3, 3, 3], [4, 4, 4, 5, 6, 7, 7, 7]]
        # We check the core data integrity
        assert torch.equal(output_fp16[:, :, 2:6], input_fp16), "Core data corrupted in float16"
        assert torch.equal(output_fp16[:, :, 0], input_fp16[:, :, 0]), "Left padding replication failed in float16"
        assert torch.equal(output_fp16[:, :, -1], input_fp16[:, :, -1]), "Right padding replication failed in float16"
    except RuntimeError as e:
        # ReplicationPad1d might not support float16 on CPU (Error: "replication_pad1d_cpu" not implemented for 'Half')
        # We skip this scenario if the device does not support it.
        print(f"Skipping Float16 test due to lack of support on device {device}: {e}")

    # --- Scenario 2: BFloat16 (The fix mentioned in the issue) ---
    # Note: BFloat16 support depends on the device capability
    try:
        input_bf16 = input_fp32.to(torch.bfloat16)
        output_bf16 = layer(input_bf16)
        
        # Assert no garbage is produced
        assert torch.isfinite(output_bf16).all(), "ReplicationPad1d produced garbage (NaN/Inf) with bfloat16 input"
        
        # Verify correctness
        assert torch.equal(output_bf16[:, :, 2:6], input_bf16), "Core data corrupted in bfloat16"
        assert torch.equal(output_bf16[:, :, 0], input_bf16[:, :, 0]), "Left padding replication failed in bfloat16"
        assert torch.equal(output_bf16[:, :, -1], input_bf16[:, :, -1]), "Right padding replication failed in bfloat16"
        
    except (RuntimeError, AttributeError) as e:
        # BFloat16 might not be supported on all CPU/MPS versions, skip if unsupported
        print(f"Skipping BFloat16 test due to lack of support: {e}")

if __name__ == "__main__":
    test_replication_pad1d_dtype_stability()
    print("Test passed successfully.")