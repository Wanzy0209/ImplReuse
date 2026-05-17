import torch
import torch.nn.functional as F
from torch import func as torch_func

def test_jacfwd_one_hot_with_compile():
    """
    Test case for Issue 160752: torch.func.jacfwd fails with one_hot and torch.compile(.., dynamic=True).
    
    This test adapts the logic of checking execution contexts (eager vs compiled),
    similar to how tf.compat.v1.local_variables_initializer handles eager vs graph modes.
    """
    MAX = 3
    BATCH = 37

    def func(x, idxs):
        return x.square() * F.one_hot(idxs, MAX)

    def jacfunc(x, idxs):
        return torch_func.jacfwd(func, argnums=(0,))(x, idxs)

    idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
    x = torch.rand((BATCH, MAX), dtype=torch.float64)

    # 1. Eager Execution
    # Corresponds to context.executing_eagerly() in TensorFlow
    print("Testing eager execution...")
    try:
        out_eager = jacfunc(x, idxs)
        # Basic shape assertion to ensure computation happened
        assert out_eager.shape == (BATCH, MAX, BATCH, MAX), f"Unexpected shape in eager mode: {out_eager.shape}"
        print("Eager execution successful.")
    except Exception as e:
        print(f"Eager execution failed: {e}")
        raise

    # 2. Compiled Execution (Dynamic=True)
    # Corresponds to Graph mode execution in TensorFlow
    print("Testing compiled execution with dynamic=True...")
    try:
        compiled_jacfunc = torch.compile(jacfunc, dynamic=True)
        out_compiled = compiled_jacfunc(x, idxs)
        
        # Verify output matches eager execution
        assert torch.allclose(out_eager, out_compiled), "Compiled output does not match eager output"
        print("Compiled execution successful.")
    except Exception as e:
        print(f"Compiled execution failed (Bug reproduced): {e}")
        # Re-raise to indicate test failure
        raise

if __name__ == "__main__":
    test_jacfwd_one_hot_with_compile()