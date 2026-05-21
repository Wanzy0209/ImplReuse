import torch

def test_torch_full_compile_float64_caching():
    """
    Test case for Issue 166253: torch.full caches fill_value with torch.compile and float64.
    Verifies that the fill_value updates correctly on subsequent calls.
    """
    def func(x):
        return torch.full((2, ), x, dtype=torch.float64)

    # Compile the function
    func_jit = torch.compile(func)

    # First call with value 5.0
    x1 = torch.tensor(5.0, dtype=torch.float64)
    res1 = func_jit(x1)
    print(f"Call 1 (5.0): {res1}")
    assert torch.all(res1 == 5.0), f"Expected [5., 5.], got {res1}"

    # Second call with value 10.0
    # Bug: Previously this would return [5., 5.] because the value was cached.
    x2 = torch.tensor(10.0, dtype=torch.float64)
    res2 = func_jit(x2)
    print(f"Call 2 (10.0): {res2}")
    assert torch.all(res2 == 10.0), f"Expected [10., 10.], got {res2}"

def test_torch_full_compile_float32():
    """
    Verify if the issue affects float32 as well.
    """
    def func(x):
        return torch.full((2, ), x, dtype=torch.float32)

    func_jit = torch.compile(func)

    x1 = torch.tensor(5.0, dtype=torch.float32)
    res1 = func_jit(x1)
    assert torch.all(res1 == 5.0)

    x2 = torch.tensor(10.0, dtype=torch.float32)
    res2 = func_jit(x2)
    assert torch.all(res2 == 10.0)

def test_torch_full_compile_int64():
    """
    Verify if the issue affects int64.
    """
    def func(x):
        return torch.full((2, ), x, dtype=torch.int64)

    func_jit = torch.compile(func)

    x1 = torch.tensor(5, dtype=torch.int64)
    res1 = func_jit(x1)
    assert torch.all(res1 == 5)

    x2 = torch.tensor(10, dtype=torch.int64)
    res2 = func_jit(x2)
    assert torch.all(res2 == 10)

if __name__ == "__main__":
    print("Running test_torch_full_compile_float64_caching...")
    test_torch_full_compile_float64_caching()
    print("PASSED\n")

    print("Running test_torch_full_compile_float32...")
    test_torch_full_compile_float32()
    print("PASSED\n")

    print("Running test_torch_full_compile_int64...")
    test_torch_full_compile_int64()
    print("PASSED\n")