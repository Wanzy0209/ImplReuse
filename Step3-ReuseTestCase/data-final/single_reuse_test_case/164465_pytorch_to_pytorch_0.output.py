import torch
import sys

def test_inductor_crash_int64_arange_max():
    """
    Regression test for Issue #164465.
    Verifies that torch.compile does not crash with int64 + arange + max operations.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # The function decorated with torch.compile
    @torch.compile
    def f(arg0_1, arg1_1):
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
        view = torch.ops.aten.view.default(embedding, [64, 3072])
        unsqueeze = torch.ops.aten.unsqueeze.default(view, 0)
        expand = torch.ops.aten.expand.default(unsqueeze, [576, -1, -1])
        view_1 = torch.ops.aten.view.default(expand, [2, 8, 36, 64, 3072])
        permute = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4])
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view_2 = torch.ops.aten.view.default(clone, [2, 18432, 3072])
        
        # The specific sequence causing the inductor crash
        # Using torch.arange (public API) equivalent to prims.iota
        iota = torch.arange(36, dtype=torch.int64, device='cuda')
        view_3 = torch.ops.aten.view.default(iota, [1, 36])
        max_1 = torch.ops.aten.max.default(view_3)
        
        return (max_1,)

    # Setup inputs
    x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
    y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)

    try:
        # Run the compiled function
        out = f(x, y)
        print("Test passed. No crash detected.")
        # Optional: Verify output shape/type if necessary, though the primary goal is crash prevention
        assert out is not None
    except Exception as e:
        print(f"Test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_inductor_crash_int64_arange_max()