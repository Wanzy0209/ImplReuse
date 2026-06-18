import torch
import numpy as np

def test_interpolate_bicubic_consistency():
    """
    Test case to verify consistency between eager and compiled modes
    for torch.nn.functional.interpolate with bicubic mode.
    Based on Issue ID: 160840
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate input data matching the bug report
    # Shape: (1, 40, 1, 1), dtype: float64
    x = np.random.uniform(0, 10, size=(1, 40, 1, 1))
    input_tensor = torch.from_numpy(x)

    # Define the function using the target API
    def interpolate_func(input):
        return torch.nn.functional.interpolate(
            input,
            size=[40, 40],
            scale_factor=None,
            mode="bicubic",
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False,
        )

    # Execute in eager mode
    eager_result = interpolate_func(input_tensor)

    # Execute in compiled mode
    compiled_func = torch.compile(interpolate_func)
    compiled_result = compiled_func(input_tensor)

    # Assert that the results are close
    # This will raise an error if the bug is present
    torch.testing.assert_close(eager_result, compiled_result)

if __name__ == "__main__":
    test_interpolate_bicubic_consistency()