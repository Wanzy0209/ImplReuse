import torch
import torch.nn.functional as F
from torch.testing import assert_close
from torch.func import jacfwd

# Constants from the bug report
MAX = 3
BATCH = 37

def func(x, idxs):
    return x.square() * F.one_hot(idxs, MAX)

def jacfunc(x, idxs):
    return jacfwd(func, argnums=(0,))(x, idxs)

def run_jacobian_test(x, idxs, use_compile=False):
    """
    Executes the Jacobian calculation in either Eager or Compiled mode.
    
    This function structure is inspired by the similar API (tf.compat.v1.train.global_step),
    which handles logic differently based on the execution context (Eager vs Graph/Session).
    Here, we adapt that pattern to PyTorch's Eager vs torch.compile contexts.
    """
    # The core logic to be tested
    def _compute(x, idxs):
        return jacfwd(func, argnums=(0,))(x, idxs)

    if use_compile:
        # Analogous to the Graph/Session execution path in TensorFlow
        # In the bug report, this path fails with dynamic=True
        compiled_fn = torch.compile(_compute, dynamic=True)
        return compiled_fn(x, idxs)
    else:
        # Analogous to the Eager execution path in TensorFlow
        # In the bug report, this path works
        return _compute(x, idxs)

def test_jacfwd_one_hot_compile():
    # Setup inputs
    idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
    x = torch.rand((BATCH, MAX), dtype=torch.float64)

    # 1. Test Eager Mode (Baseline)
    # This corresponds to the 'if context.executing_eagerly()' branch in the similar API
    print("Testing Eager mode...")
    try:
        out_eager = run_jacobian_test(x, idxs, use_compile=False)
        print("Eager mode execution successful.")
    except Exception as e:
        print(f"Eager mode failed: {e}")
        raise

    # 2. Test Compiled Mode (Bug Reproduction)
    # This corresponds to the Graph/Session execution path
    print("Testing Compiled mode (dynamic=True)...")
    try:
        out_compiled = run_jacobian_test(x, idxs, use_compile=True)
        
        # If the bug is fixed, outputs should match
        assert_close(out_eager, out_compiled, rtol=1e-4, atol=1e-4)
        print("Compiled mode execution successful and matches eager output.")
        
    except Exception as e:
        print(f"Compiled mode failed (Bug Reproduced): {e}")
        # Re-raise to indicate test failure due to the bug
        raise

if __name__ == "__main__":
    test_jacfwd_one_hot_compile()