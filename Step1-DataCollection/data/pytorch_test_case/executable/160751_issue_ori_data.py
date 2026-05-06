import torch

def func():
    a = torch.tensor([1.0, -2.0], device="cuda")
    result = torch.all(a > 0)
    assert result, "should throw"
    torch.cuda.synchronize()
    print("should not run")


def test_fn():
    torch._dynamo.reset()
    f_c = torch.compile(func, backend="aot_eager")
    f_c()

test_fn()