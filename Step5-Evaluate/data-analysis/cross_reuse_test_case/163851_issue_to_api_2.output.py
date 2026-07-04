import torch
import pytest

def test_matrix_rank_nan_handling():
    """
    Test case for torch.linalg.matrix_rank (the successor to torch.matrix_rank)
    adapted from the logic of the grid_sampler_3d NaN issue.
    
    This test verifies that the MPS backend handles NaN values in the input
    consistently with the CPU backend.
    """
    # Create a matrix with a NaN value, similar to how the original issue
    # created a grid with a NaN value.
    input_tensor = torch.ones(3, 3)
    input_tensor[1, 1] = float('nan')

    # Compute rank on CPU
    rank_cpu = torch.linalg.matrix_rank(input_tensor)

    # Compute rank on MPS if available
    if torch.backends.mps.is_available():
        rank_mps = torch.linalg.matrix_rank(input_tensor.to("mps"))

        print(f"CPU Rank: {rank_cpu}")
        print(f"MPS Rank: {rank_mps}")

        # The original bug (grid_sampler_3d) showed that MPS returned a valid number (1.0)
        # instead of NaN when the input contained NaN.
        # We assert that if CPU returns NaN, MPS must also return NaN.
        if torch.isnan(rank_cpu):
            assert torch.isnan(rank_mps), (
                f"MPS returned {rank_mps} but CPU returned NaN. "
                "MPS should propagate NaN like CPU."
            )
        else:
            assert rank_cpu == rank_mps, (
                f"MPS returned {rank_mps} but CPU returned {rank_cpu}. "
                "Results should match."
            )
    else:
        pytest.skip("MPS backend not available")

if __name__ == "__main__":
    try:
        test_matrix_rank_nan_handling()
    except pytest.Skipped as e:
        print(f"Skipped: {e}")