import torch

def test_inductor_crash_int64_arange_max():
    """
    Test case for Issue 164465: Inductor crash with int64 + arange + max.
    This test preserves the original bug reproduction logic and leverages 
    torch.randn for input generation as identified by the similarity analysis.
    """
    # Ensure CUDA is available as the bug is specific to it
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

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
        
        # Critical section: int64 iota (arange) + view + max
        iota = torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, device = 'cuda', requires_grad = False)
        view_3 = torch.ops.aten.view.default(iota, [1, 36])
        max_1 = torch.ops.aten.max.default(view_3)
        return (max_1,)

    # Leverage torch.randn (Similar API) for input generation
    x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
    y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)

    # Run the test
    out = f(x, y)
    
    # Verify the result (max of 0..35 is 35)
    assert out[0].item() == 35, f"Expected max value 35, got {out[0].item()}"

if __name__ == "__main__":
    test_inductor_crash_int64_arange_max()