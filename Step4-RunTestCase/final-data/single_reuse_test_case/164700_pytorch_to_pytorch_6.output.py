import os
import torch

# Reproduce the environment settings from the bug report
os.environ["TORCH_LOGS"] = "output_code"

device = "cuda"

def f(x, y):
    """
    Adapted function that includes the problematic unsqueeze and cat pattern
    from the bug report, but utilizes torch.lobpcg as the primary operation.
    """
    # --- The problematic pattern from the bug report ---
    # This pattern involves interleaving unsqueeze (via None indexing) and torch.cat
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    # Combine the tensors
    # Shape of 'combined' will be (1, 32, 1) based on the input shapes
    combined = torch.cat([x2, y3], dim=1)

    # --- Adaptation for torch.lobpcg ---
    # torch.lobpcg requires a symmetric positive definite matrix A and an initial guess X.
    # We use the 'combined' tensor (after reshaping and type casting) as the initial guess X.
    
    # combined is (1, 32, 1). Squeeze to (32, 1) for X.
    # lobpcg requires floating point inputs, so we cast from int.
    X = combined.squeeze(0).to(torch.float32)
    
    # Create a dummy symmetric positive definite matrix A (32x32)
    A = torch.eye(32, device=device, dtype=torch.float32)

    # Call the similar API: torch.lobpcg
    # We are testing if torch.compile can handle the pattern leading up to this call
    eigenvalues, _ = torch.lobpcg(A, X=X)
    
    return eigenvalues

# Inputs from the original bug report
x = torch.zeros(1, 32, dtype=torch.int64, device=device)
y = torch.zeros(1, dtype=torch.int32, device=device)

# Test 1: Eager execution (to verify logic correctness)
try:
    result_eager = f(x, y)
    print("Eager execution succeeded.")
except Exception as e:
    print(f"Eager execution failed: {e}")

# Test 2: Compiled execution (The scenario that caused the crash in the original bug)
# We replace the original call site logic with the adapted function containing torch.lobpcg
try:
    compiled_f = torch.compile(f)
    result_compiled = compiled_f(x, y)
    print("Compiled execution succeeded.")
    
    # Verify results match
    assert torch.allclose(result_eager, result_compiled), "Results differ between eager and compiled"
    print("Results match.")
except Exception as e:
    print(f"Compiled execution failed: {e}")