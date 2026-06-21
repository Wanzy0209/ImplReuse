import torch

# Handle environments where torch.compile is not available (e.g., PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    torch.compile = lambda func: func

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
    iota = torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, device = 'cuda', requires_grad = False)
    view_3 = torch.ops.aten.view.default(iota, [1, 36])
    # Adaptation: Replace torch.ops.aten.max.default with torch.all
    all_1 = torch.all(view_3)
    return (all_1,)


if __name__ == "__main__":
    x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
    y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
    
    # Run the compiled function
    out = f(x, y)
    
    # Verify the result. iota starts at 0, so all() should be False.
    assert out[0].item() == False
    print("Test passed.")