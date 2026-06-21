import torch

# Determine the device to use. Fallback to CPU if XPU is not available.
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"
    print("Warning: XPU device not found. Using CPU instead.")

def lobpcg_func(A, k):
    # torch.lobpcg requires a symmetric positive definite matrix A and an initial guess X
    # We generate X inside the function to keep the signature simple
    X = torch.randn(A.shape[0], k, device=A.device)
    e, v = torch.lobpcg(A, k, X=X)
    return e, v

# Setup inputs
# Create a symmetric positive definite matrix of size 128x128 to match the scale of the original bug
n = 128
k = 3
A_raw = torch.randn(n, n).to(device)
A = A_raw @ A_raw.T + torch.eye(n).to(device)

# Test eager mode
try:
    e_eager, v_eager = lobpcg_func(A, k)
    print(f"eager mode passed on {device}")
except Exception as e:
    print(f"eager mode failed: {e}")

# Test compiled mode
lobpcg_func_compiled = torch.compile(lobpcg_func)
try:
    e_compiled, v_compiled = lobpcg_func_compiled(A, k)
    print(f"torch.compile passed on {device}")
    # Basic assertion to verify output shape if it doesn't crash
    assert e_compiled.shape == (k,)
except Exception as e:
    print(f"torch.compile failed: {e}")