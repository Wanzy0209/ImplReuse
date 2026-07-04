import torch

def test_rand_like_large_tensor():
    # Test torch.rand_like with large tensors (>4GB) on MPS.
    # The original bug (torch.ones) manifested as zeros at the end of the buffer
    # due to fillBuffer failing. We check if rand_like (using RNG kernels)
    # correctly initializes the entire buffer.

    # To exceed 4GB with float32 (default for rand_like), we need > 2^30 elements.
    # Shape: (2, (1 << 29) + 5) -> approx 1.07 billion elements -> ~4.3 GB
    shape = (2, (1 << 29) + 5)

    try:
        # Create a prototype tensor. Using empty to avoid the 'ones' bug during setup.
        prototype = torch.empty(shape, device='mps')
    except (NotImplementedError, RuntimeError) as e:
        # Handle cases where MPS is not available or the specific operator is missing
        print(f"Skipping test: MPS backend is not supported or missing required operators. Error: {e}")
        return

    # Generate random tensor using the similar API
    a = torch.rand_like(prototype)

    # Check specific indices at the end of the buffer, similar to the original test case.
    # Original bug: a[1, -2] was 0.
    val_single = a[1, -2].item()
    val_slice = a[:, -2]

    print(f"Value at [1, -2]: {val_single}")
    print(f"Values at [:, -2]: {val_slice}")

    # Assert values are valid random numbers (strictly between 0 and 1).
    # This ensures the buffer was fully written to and not left as zeros (uninitialized).
    assert 0 < val_single < 1, f"Expected random value in (0, 1), got {val_single}"
    assert (val_slice > 0).all() and (val_slice < 1).all(), \
        f"Expected random values in (0, 1), got {val_slice}"

if __name__ == "__main__":
    test_rand_like_large_tensor()