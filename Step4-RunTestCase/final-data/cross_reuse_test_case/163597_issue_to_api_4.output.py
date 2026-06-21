import torch
import torch.nn.functional as F

def test_sdpa_mps_with_cropping_pattern():
    """
    Test case for Issue 163597: SDPA MPS regression on non-contiguous tensors.
    
    This test leverages the logic pattern of tf.keras.layers.Cropping3D (slicing 
    spatial dimensions of 5D data) to generate non-contiguous tensor views. 
    These views are then reshaped and transposed to trigger the specific 
    non-contiguous tensor handling bug in torch.nn.functional.scaled_dot_product_attention 
    on the MPS device.
    """
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    # 1. Mimic tf.keras.layers.Cropping3D input creation
    # Cropping3D typically operates on 5D data: (Batch, Depth, Height, Width, Channels)
    batch_size = 2
    depth, height, width = 10, 10, 10
    channels = 64
    
    # Create base tensors on MPS and CPU
    x_mps = torch.randn(batch_size, depth, height, width, channels, device="mps")
    x_cpu = x_mps.cpu()

    # 2. Apply Cropping3D logic (slicing) to create non-contiguous views
    # We crop 2 units from each spatial dimension (Depth, Height, Width)
    # This slicing operation creates a view (non-contiguous tensor) similar to 
    # the output of a Cropping3D layer.
    crop_d, crop_h, crop_w = 2, 2, 2
    cropped_mps = x_mps[:, crop_d:-crop_d, crop_h:-crop_h, crop_w:-crop_w, :]
    cropped_cpu = x_cpu[:, crop_d:-crop_d, crop_h:-crop_h, crop_w:-crop_w, :]

    # 3. Reshape for SDPA
    # Target shape for SDPA: (Batch, SeqLen, NumHeads, HeadDim)
    # We flatten the remaining spatial dimensions into the sequence length.
    num_heads = 4
    spatial_dim = (depth - 2*crop_d) * (height - 2*crop_h) * (width - 2*crop_w) # 6*6*6 = 216
    head_dim = channels
    
    # Reshape and Transpose to match the non-contiguous pattern in the original bug report.
    # The original bug was triggered by tensors transposed from (Batch, Seq, Heads, HeadDim)
    # to (Batch, Heads, Seq, HeadDim).
    # Note: Resaping a non-contiguous slice might result in a copy or a view depending on 
    # implementation, but the subsequent transpose guarantees a non-contiguous tensor.
    
    q_mps = cropped_mps.reshape(batch_size, num_heads, spatial_dim // num_heads, head_dim).transpose(1, 2)
    k_mps = cropped_mps.reshape(batch_size, num_heads, spatial_dim // num_heads, head_dim).transpose(1, 2)
    v_mps = cropped_mps.reshape(batch_size, num_heads, spatial_dim // num_heads, head_dim).transpose(1, 2)

    q_cpu = cropped_cpu.reshape(batch_size, num_heads, spatial_dim // num_heads, head_dim).transpose(1, 2)
    k_cpu = cropped_cpu.reshape(batch_size, num_heads, spatial_dim // num_heads, head_dim).transpose(1, 2)
    v_cpu = cropped_cpu.reshape(batch_size, num_heads, spatial_dim // num_heads, head_dim).transpose(1, 2)

    # 4. Run Scaled Dot Product Attention
    # The bug affects the fast MPS implementation when inputs are non-contiguous.
    out_mps = F.scaled_dot_product_attention(q_mps, k_mps, v_mps)
    out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

    # 5. Assert correctness
    # Compare MPS output against CPU reference.
    # If the regression is present, the MPS output will differ significantly from CPU.
    diff = torch.norm(out_mps.cpu() - out_cpu)
    print(f"Norm difference (MPS vs CPU): {diff.item():.6f}")
    
    # Assert that the difference is within acceptable floating point tolerance.
    # A large difference indicates the non-contiguous tensor bug is active.
    assert torch.allclose(out_mps.cpu(), out_cpu, atol=1e-4), \
        f"SDPA MPS regression detected. Difference: {diff.item():.6f}"

if __name__ == "__main__":
    test_sdpa_mps_with_cropping_pattern()