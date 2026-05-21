import torch


def func_nojit(x):
    return torch.is_complex(x)


func_jit = torch.compile(func_nojit)

# Test with float64 (from the original bug report) and complex128
x1 = torch.tensor(5.0, dtype=torch.float64)
x2 = torch.tensor(1.0 + 2.0j, dtype=torch.complex128)

for func in [func_nojit, func_jit]:
    print(f"Testing {func.__name__}")
    result1 = func(x1)
    result2 = func(x2)
    print(f"  is_complex(float64): {result1}")
    print(f"  is_complex(complex128): {result2}")
    
    # Assertions to verify correct behavior
    assert result1 == False, f"Expected False for float64, got {result1}"
    assert result2 == True, f"Expected True for complex128, got {result2}"