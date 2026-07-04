import torch

def test_inductor_sort_int64():
    """
    Test case for torch.sort based on the crash pattern observed with torch.max (Issue 164465).
    The original issue involved an Inductor crash when processing int64 tensors (generated via iota/arange)
    through a reduction operation (max) on CUDA. This test checks if the similar API (torch.sort)
    exhibits the same behavior or handles the int64 type correctly under torch.compile.
    """
    # Check for CUDA availability as the original bug was specific to CUDA/Triton
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Check for torch.compile availability (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch >= 2.0), skipping test.")
        return

    # Define the function using the similar API (torch.sort)
    @torch.compile
    def f_sort(x):
        # Mimic the structure of the original repro: int64 tensor -> view -> operation
        # Original: view_3 = torch.ops.aten.view.default(iota, [1, 36])
        #          max_1 = torch.ops.aten.max.default(view_3)
        view = torch.ops.aten.view.default(x, [1, 36])
        
        # Apply the similar API (sort) instead of max
        # We sort along the last dimension
        sorted_vals, sorted_indices = torch.sort(view, dim=-1)
        return sorted_vals, sorted_indices

    # Reproduce the input setup: int64 tensor on CUDA
    # Original used: torch.ops.prims.iota.default(..., dtype=torch.int64, device='cuda')
    # torch.arange is the standard user-facing equivalent for iota
    x = torch.arange(36, dtype=torch.int64, device='cuda')

    try:
        # Run the compiled function
        out = f_sort(x)

        # Verify against eager execution to ensure correctness
        expected_vals, expected_indices = torch.sort(x.view(1, 36), dim=-1)

        assert torch.equal(out[0], expected_vals), "Values mismatch between compiled and eager"
        assert torch.equal(out[1], expected_indices), "Indices mismatch between compiled and eager"

        print("Test passed: torch.sort handles int64 correctly under torch.compile.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_inductor_sort_int64()