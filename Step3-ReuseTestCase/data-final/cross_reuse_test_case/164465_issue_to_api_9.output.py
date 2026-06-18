import torch

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
    
    # The bug reproduction logic involves int64 generation via iota, view, and a reduction.
    # We replace the 'max' reduction with the similar 'var_mean' API.
    iota = torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, device = 'cuda', requires_grad = False)
    view_3 = torch.ops.aten.view.default(iota, [1, 36])
    
    # Original: max_1 = torch.ops.aten.max.default(view_3)
    # Similar API: var_mean
    var, mean = torch.ops.aten.var_mean.default(view_3)
    
    return (var, mean)


if torch.cuda.is_available():
    x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
    y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
    
    # Run the test to check for inductor crash
    try:
        out = f(x, y)
        print("Test passed. Output:", out)
    except Exception as e:
        print(f"Test failed with error: {e}")
else:
    print("CUDA not available, skipping test.")