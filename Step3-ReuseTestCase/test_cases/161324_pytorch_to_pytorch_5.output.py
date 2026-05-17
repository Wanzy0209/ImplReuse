import torch
import torch.nn.functional as F

def test_mse_loss_with_2d_tensor_views():
    """
    Test case adapted from Issue 161324.
    Verifies that torch.nn.functional.mse_loss handles 2D tensor views
    correctly without data inconsistencies, similar to the conditions
    that caused failures in batch_isend_irecv.
    """
    # Setup parameters mimicking the bug report
    batch_size = 4
    total_columns = 10
    # Define split offsets to create non-contiguous views (slices)
    split_offsets = [0, 3, 5, 10]

    # Create base tensors
    input_tensor = torch.randn(batch_size, total_columns)
    target_tensor = torch.randn(batch_size, total_columns)

    # Create 2D tensor views (slices) mimicking the bug report's setup
    # These views are non-contiguous in memory
    input_view1 = input_tensor[:, split_offsets[0]:split_offsets[1]]
    input_view2 = input_tensor[:, split_offsets[2]:split_offsets[3]]

    target_view1 = target_tensor[:, split_offsets[0]:split_offsets[1]]
    target_view2 = target_tensor[:, split_offsets[2]:split_offsets[3]]

    # Call the similar API: torch.nn.functional.mse_loss
    # Calculate loss on the views
    loss_view1 = F.mse_loss(input_view1, target_view1)
    loss_view2 = F.mse_loss(input_view2, target_view2)

    # Verification: Compare against contiguous versions to ensure no data inconsistency
    # If the API handles views correctly, the results should match exactly.
    input_cont1 = input_view1.contiguous()
    target_cont1 = target_view1.contiguous()
    input_cont2 = input_view2.contiguous()
    target_cont2 = target_view2.contiguous()

    loss_cont1 = F.mse_loss(input_cont1, target_cont1)
    loss_cont2 = F.mse_loss(input_cont2, target_cont2)

    # Assertions to check for data consistency
    assert torch.allclose(loss_view1, loss_cont1), \
        f"Data inconsistency detected in mse_loss for view 1: {loss_view1} vs {loss_cont1}"
    assert torch.allclose(loss_view2, loss_cont2), \
        f"Data inconsistency detected in mse_loss for view 2: {loss_view2} vs {loss_cont2}"

    print("Test passed: mse_loss handles 2D tensor views correctly.")

if __name__ == "__main__":
    test_mse_loss_with_2d_tensor_views()