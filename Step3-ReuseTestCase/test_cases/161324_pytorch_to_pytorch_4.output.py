import torch

def test_std_mean_with_2d_tensor_views():
    """
    Test case adapted from Issue 161324.
    Original issue: Data inconsistencies when using batch_isend_irecv with 2D tensor views.
    Adaptation: Verifying that torch.std_mean handles 2D tensor views (slices) correctly
    without data inconsistencies.
    """
    # Setup parameters mimicking the bug report scenario
    batch_size = 8
    total_columns = 16
    # Define split offsets for slicing columns
    split_offsets = [0, 4, 8, 16]

    # Create a random 2D tensor
    local_tensor = torch.randn(batch_size, total_columns)

    # Create 2D tensor views (slices) as done in the bug report
    # These views are non-contiguous
    view1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
    view2 = local_tensor[:, split_offsets[2]:split_offsets[3]]

    # Call the similar API: torch.std_mean on the views
    # We calculate stats over the whole view (dim=None)
    std_v1, mean_v1 = torch.std_mean(view1)
    std_v2, mean_v2 = torch.std_mean(view2)

    # Calculate ground truth using contiguous clones of the data
    # If the API fails silently with views (like the distributed bug),
    # these values might differ.
    clone1 = view1.clone()
    clone2 = view2.clone()

    std_c1, mean_c1 = torch.std_mean(clone1)
    std_c2, mean_c2 = torch.std_mean(clone2)

    # Verify consistency
    assert torch.allclose(std_v1, std_c1), f"Std mismatch for view 1: {std_v1} vs {std_c1}"
    assert torch.allclose(mean_v1, mean_c1), f"Mean mismatch for view 1: {mean_v1} vs {mean_c1}"
    assert torch.allclose(std_v2, std_c2), f"Std mismatch for view 2: {std_v2} vs {std_c2}"
    assert torch.allclose(mean_v2, mean_c2), f"Mean mismatch for view 2: {mean_v2} vs {mean_c2}"

    print("Test passed: torch.std_mean works correctly with 2D tensor views.")

if __name__ == "__main__":
    test_std_mean_with_2d_tensor_views()