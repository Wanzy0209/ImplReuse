buckets = torch.arange(10, device="cuda")
x = torch.tensor([2.5, 4.5, 5.5], device = "cuda")

def test_fn(x):
    return torch.bucketize(x, buckets[1:])

compiled_fn = torch.compile(test_fn)
out_not_compiled = test_fn(x)
out_compiled = compiled_fn(x)
print(out_not_compiled)
print(out_compiled)