import torch

def func():
    # Adapted to use torch.prod (the similar API) instead of torch.all
    a = torch.tensor([1.0, -2.0], device="cuda")
    result = torch.prod(a)
    # This assertion will fail because the product is -2.0
    assert result > 0, "should throw"
    torch.cuda.synchronize()
    print("should not run")

def test_fn():
    # Check if _dynamo exists before calling reset to handle environment differences
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    # The bug is specific to the aot_eager backend of torch.compile
    f_c = torch.compile(func, backend="aot_eager")
    f_c()

if __name__ == "__main__":
    test_fn()