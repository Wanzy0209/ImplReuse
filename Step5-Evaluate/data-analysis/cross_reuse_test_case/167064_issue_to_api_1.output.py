import torch
import tensorflow as tf

def test_gather_does_not_modify_global_state():
    """
    Test that tf.compat.v1.gather does not have unintended side effects on global state.
    This mirrors the PyTorch issue where torch.compile inadvertently changed
    global distribution validation settings.
    """
    # Set a known global state (random seed)
    tf.random.set_seed(42)
    
    # Create dummy data for gather
    params = tf.constant([[1, 2, 3], [4, 5, 6]])
    indices = tf.constant([0, 1])
    
    # Call the similar API (tf.compat.v1.gather)
    # In the PyTorch issue, the redundant code was executed.
    # Here we execute the op to check for side effects.
    _ = tf.compat.v1.gather(params, indices)

    # Verify that the global state (random seed behavior) remains consistent.
    # If the op had side effects (like re-seeding or changing global config),
    # this sequence might differ or fail.
    val1 = tf.random.uniform(shape=[1])
    val2 = tf.random.uniform(shape=[1])
    
    # We expect val1 and val2 to be different because the seed was set once,
    # and the sequence of random calls should progress.
    # If gather reset the seed or state unexpectedly, this might fail or behave differently.
    # However, the most direct check is ensuring the operation completes without
    # raising errors related to global state corruption, and that subsequent
    # random operations are still deterministic based on the initial seed.
    
    # Re-seed to check determinism
    tf.random.set_seed(42)
    val1_check = tf.random.uniform(shape=[1])
    
    assert tf.reduce_all(val1 == val1_check).numpy(), \
        "Global random state was likely modified by tf.compat.v1.gather"

if __name__ == "__main__":
    test_gather_does_not_modify_global_state()
    print("Test passed: tf.compat.v1.gather preserves global state.")