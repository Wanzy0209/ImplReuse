import tensorflow as tf

# This test case is generated based on the PyTorch MPS backend availability issue (Issue ID: 164943).
# The issue highlights the importance of correctly checking the environment or state
# (OS version in PyTorch, distribution context in TensorFlow) before performing operations.
# We test tf.distribute.in_cross_replica_context to ensure it accurately reports the current state.

def test_cross_replica_context_behavior():
    # 1. Check default state (outside strategy)
    # Analogous to checking MPS availability on an unsupported OS (should be False/Unavailable)
    assert not tf.distribute.in_cross_replica_context(), \
        "Expected to be outside cross-replica context initially."

    # 2. Check state inside strategy.scope()
    # Analogous to checking MPS availability on a supported OS (should be True/Available)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        assert tf.distribute.in_cross_replica_context(), \
            "Expected to be in cross-replica context inside strategy.scope()."

        # 3. Check state inside strategy.run()
        # Analogous to a specific sub-context or mode where the condition changes
        def step_fn():
            assert not tf.distribute.in_cross_replica_context(), \
                "Expected to be in replica context, not cross-replica context, inside strategy.run()."

        strategy.run(step_fn)

if __name__ == "__main__":
    test_cross_replica_context_behavior()
    print("Test passed successfully.")