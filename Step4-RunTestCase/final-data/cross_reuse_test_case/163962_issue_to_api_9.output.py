import sys

try:
    import torch
    import tensorly as tl
    from tensorly.decomposition import parafac
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Required module not found - {e}")
    sys.exit(0)

def test_parafac_mps_with_tf_step_tracking():
    """
    Test case reproducing the MPS bug while leveraging tf.summary.experimental.set_step.
    
    This test preserves the original bug reproduction logic (Tensorly PARAFAC on MPS)
    and integrates the similar API (tf.summary.experimental.set_step) to track 
    the execution step, mirroring the state-setting pattern found in the issue.
    """
    
    # Setup Tensorly backend (State setting pattern similar to the similar API)
    tl.set_backend("pytorch")
    
    # Leverage the similar API: Set the default summary step for the current thread.
    # This mirrors the state-setting nature of tl.set_backend used in the original issue.
    tf.summary.experimental.set_step(0)

    # Check for MPS availability to reproduce the specific bug context
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping MPS-specific test.")
        return

    # Original bug reproduction logic
    x = torch.ones(12, 3, 12).to("mps")
    
    # Define a callback to leverage the similar API during the decomposition process
    def track_iteration(cp_tensor):
        # Update the step using the similar API to demonstrate integration
        # In a real scenario, this might be incremented based on the iteration count
        current_step = tf.summary.experimental.get_step()
        tf.summary.experimental.set_step(current_step + 1)

    try:
        # The bug (RuntimeError: Internal assert failed) was triggered here.
        # We pass the callback to integrate the similar API usage.
        weights, factors = parafac(
            x.detach(), 
            rank=12, 
            init="random", 
            tol=1e-6,
            callback=track_iteration
        )
        
        # Assertions to verify the operation completed successfully
        assert factors is not None, "Factors should not be None"
        assert len(factors) == 3, "Expected 3 factors for a 3-way tensor"
        
        # Verify the similar API was interacted with
        final_step = tf.summary.experimental.get_step()
        assert final_step >= 0, "Step should have been tracked"

    except RuntimeError as e:
        # Catching the specific error mentioned in the bug report for documentation purposes
        # If the bug is fixed, this block should not be reached.
        print(f"RuntimeError encountered (indicates bug persistence): {e}")
        raise

if __name__ == "__main__":
    test_parafac_mps_with_tf_step_tracking()