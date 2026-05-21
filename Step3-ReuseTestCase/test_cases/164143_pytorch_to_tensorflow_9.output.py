import torch
import tensorflow as tf
from tensorflow.python.framework import ops
from contextlib import nullcontext

def test_name_scope_skip_on_eager():
    """
    Adapted from Issue 164143: DebugMode silently disables torch.compile.
    
    In PyTorch, the presence of a non-infra dispatch mode (DebugMode) causes 
    torch.compile to silently skip compilation.
    
    In TensorFlow, the internal ops.name_scope has a similar behavior where
    it can silently skip applying the scope (returning a NullContextManager)
    when executing eagerly, controlled by the 'skip_on_eager' flag.
    
    This test verifies that the 'skip_on_eager' parameter correctly controls
    this silent skipping behavior.
    """
    
    # Ensure we are in eager mode (default in TF 2.x)
    assert tf.executing_eagerly()

    # Case 1: skip_on_eager=True (Default behavior)
    # This mimics the "silent disabling" seen in the PyTorch bug.
    # The API should return a NullContextManager, effectively skipping the scope.
    print("Testing skip_on_eager=True (Default)...")
    ctx_skip = ops.name_scope("test_scope", skip_on_eager=True)
    
    # Verify that the returned context manager is a NullContext (or behaves like one)
    # Note: Depending on TF version, the exact class might vary, but the intent is NullContext.
    # We check if it is the standard library nullcontext or a TF internal equivalent.
    is_null_context = isinstance(ctx_skip, type(nullcontext()))
    
    if is_null_context:
        print("SKIPPING: Scope is skipped (NullContext returned) as expected.")
    else:
        # If it's not the exact nullcontext type, we verify behavior by checking if it enters/exits without error
        # and without modifying the name scope (though checking name scope in eager is tricky).
        print("INFO: Context manager type is not standard nullcontext, verifying behavior...")
    
    with ctx_skip:
        # In this block, the scope should be skipped.
        pass

    # Case 2: skip_on_eager=False
    # This mimics the desired behavior where the feature (scoping) works despite the mode (eager).
    print("\nTesting skip_on_eager=False...")
    ctx_active = ops.name_scope("test_scope", skip_on_eager=False)
    
    # This should return a valid NameScope context manager, not a NullContext.
    is_not_null_context = not isinstance(ctx_active, type(nullcontext()))
    
    if is_not_null_context:
        print("ACTIVE: Scope is active (NameScope returned) as expected.")
        
    with ctx_active:
        # In this block, the scope should be active.
        pass

    # Assertions to verify the logic
    # We expect the default behavior to be a "skip" (NullContext) in eager mode
    # similar to how PyTorch's compile skips in DebugMode.
    assert isinstance(ctx_skip, type(nullcontext())), \
        "ops.name_scope with skip_on_eager=True should return NullContext in eager mode"
        
    assert not isinstance(ctx_active, type(nullcontext())), \
        "ops.name_scope with skip_on_eager=False should return a valid scope context in eager mode"

if __name__ == "__main__":
    test_name_scope_skip_on_eager()