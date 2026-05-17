import torch

print(torch.__version__)

# Recreate tensors from the original bug report
tensor1 = torch.randint(
    low=-100,
    high=100,
    size=(9, 3, 7),
    dtype=torch.int16
)
tensor2 = torch.randint(
    low=0,
    high=2,
    size=(1, 6, 4, 8),
    dtype=torch.bool
)

# Recreate the "garbage" arguments from the original bug report
huge_int = 154691921484029491302139942063978250367
empty_list = []
empty_tuple = ()

# Adaptation 1: Pass the garbage arguments directly to affine_grid
# Original: MaxUnpool2d([], huge_int, ())
# Adapted: affine_grid([], huge_int, ())
print("Testing with garbage arguments (empty list, huge int, empty tuple)...")
try:
    r = torch.nn.functional.affine_grid(empty_list, huge_int, empty_tuple)
    print("Result:", r)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")

# Adaptation 2: Pass the specific tensor types with the huge integer as size
# Original: r1(tensor1, tensor2)
# Adapted: affine_grid(tensor1, huge_int)
print("\nTesting with tensor1 (int16) and huge int as size...")
try:
    r = torch.nn.functional.affine_grid(tensor1, huge_int)
    print("Result:", r)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")

# Adaptation 3: Pass tensor2 (bool) with huge int as size
print("\nTesting with tensor2 (bool) and huge int as size...")
try:
    r = torch.nn.functional.affine_grid(tensor2, huge_int)
    print("Result:", r)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")