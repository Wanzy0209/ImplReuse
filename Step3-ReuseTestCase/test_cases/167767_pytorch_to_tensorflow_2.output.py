import torch
import tensorflow as tf

# The original bug report describes torch.clamp failing to enforce a minimum value (1e-7) 
# on a tensor of zeros. The values remained 0 instead of being clamped to 1e-7.
#
# This test adapts the logic to tf.raw_ops.TruncatedNormal. While TruncatedNormal is a 
# generative operation rather than a manipulation operation, it enforces strict bounds 
# on its output: values are constrained to [mean - 2*stddev, mean + 2*stddev].
#
# We verify that the generated values strictly respect the lower bound, mirroring the 
# verification logic of the original clamp bug.

def test_truncated_normal_bounds():
    # Setup parameters to define a specific lower bound
    mean = 0.0
    stddev = 0.5
    # The theoretical lower bound for TruncatedNormal is mean - 2 * stddev
    expected_min = mean - 2 * stddev  # Should be -1.0

    # Generate values using the raw op
    # We use a fixed seed to ensure reproducibility
    tensor = tf.raw_ops.TruncatedNormal(
        shape=[100],
        dtype=tf.float32,
        mean=mean,
        stddev=stddev,
        seed=42,
        seed2=0
    )

    print(f"Generated Tensor (first 5 elements): {tensor[:5]}")
    print(f"Expected Lower Bound: {expected_min}")

    # Verify the lower bound is respected
    # In the PyTorch bug, values < min remained unchanged (incorrect).
    # Here we verify that no values fall below the calculated lower bound.
    all_above_min = tf.reduce_all(tensor >= expected_min)

    print(f"All values respect lower bound: {all_above_min}")

    # Assertion to catch potential incorrect behavior (similar to the original bug)
    assert all_above_min.numpy(), f"Bug detected: Values found below the expected lower bound of {expected_min}"

if __name__ == "__main__":
    test_truncated_normal_bounds()