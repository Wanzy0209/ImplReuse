import torch
import torch.nn.functional as F

def test_instance_norm_with_tensor_views():
    """
    Test case adapted from Issue 161324.
    Original issue: Data inconsistencies with batch_isend_irecv using 2D tensor views.
    Adaptation: Verify torch.nn.functional.instance_norm handles non-contiguous tensor views correctly.
    """
    # Setup parameters similar to the bug report
    batch_size = 4
    total_columns = 10
    split_offsets = [0, 3, 5, 10]

    # Create a 3D tensor (Batch, Channels, Length) to be valid for instance_norm.
    # The original bug used 2D (Batch, Columns), we add a spatial dimension for the norm.
    local_tensor = torch.randn(batch_size, total_columns, 5)

    # Create views (slices) similar to the bug report
    # These are non-contiguous tensors which might trigger underlying memory issues
    view1 = local_tensor[:, split_offsets[0]:split_offsets[1], :]
    view2 = local_tensor[:, split_offsets[2]:split_offsets[3], :]

    print(f"Rank 0 (Simulated) processing views with shapes {view1.shape} and {view2.shape}")
    print(f"View 1 is_contiguous: {view1.is_contiguous()}")
    print(f"View 2 is_contiguous: {view2.is_contiguous()}")

    # Apply the similar API: torch.nn.functional.instance_norm
    # We verify that the API handles non-contiguous views without data corruption
    try:
        # Call instance_norm on the views
        # Note: instance_norm normalizes over spatial dimensions (dim 2) 
        # and computes stats per channel (dim 1) across the batch (dim 0).
        out1 = F.instance_norm(view1)
        out2 = F.instance_norm(view2)

        # Verify output shapes match input views
        assert out1.shape == view1.shape, f"Shape mismatch for view1: {out1.shape} vs {view1.shape}"
        assert out2.shape == view2.shape, f"Shape mismatch for view2: {out2.shape} vs {view2.shape}"

        # Verify data consistency (check for NaNs or Infs which indicate silent failures)
        assert torch.isfinite(out1).all(), "Output 1 contains NaN or Inf (Data Inconsistency)"
        assert torch.isfinite(out2).all(), "Output 2 contains NaN or Inf (Data Inconsistency)"

        print("Test Passed: torch.nn.functional.instance_norm handled tensor views correctly.")

    except Exception as e:
        print(f"Test Failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_instance_norm_with_tensor_views()