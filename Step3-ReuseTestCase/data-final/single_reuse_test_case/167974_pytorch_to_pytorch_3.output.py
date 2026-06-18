import torch
import torch.nn as nn

def test_conv1d_with_2d_input():
    """
    Test case for torch.nn.Conv1d adapted from the EmbeddingBag bug report.
    The original bug involved incorrect handling of 2D inputs with a specific flag.
    This test verifies that Conv1d handles the adapted 2D input correctly.
    """
    # Input from the original bug report
    # Shape: (2, 4) -> Batch size 2, Length 4
    # Note: EmbeddingBag accepts Long (indices), Conv1d accepts Float
    input_data = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.float)

    # Adapt input for Conv1d: (Batch, Channels, Length)
    # Conv1d requires a 3D input, so we unsqueeze to add a channel dimension.
    input_conv = input_data.unsqueeze(1)

    # Initialize Conv1d
    # in_channels=1 (to match the unsqueezed input)
    # out_channels=3 (matching the EmbeddingBag output dim in the bug report)
    conv1d = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=1)

    # Run the model
    output = conv1d(input_conv)

    # Verify output shape
    # Expected: (Batch=2, Channels=3, Length=4)
    expected_shape = (2, 3, 4)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print("Test passed.")

if __name__ == "__main__":
    test_conv1d_with_2d_input()