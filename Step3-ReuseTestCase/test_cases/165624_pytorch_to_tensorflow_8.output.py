import tensorflow as tf

def test_duplicate_scope_application():
    """
    Adapts the logic from PyTorch issue 165624 to TensorFlow.
    
    The original bug involved a merge mistake where a custom pre-pass 
    (joint_custom_pre_pass) was executed twice. This test simulates 
    that structural pattern using tf.keras.name_scope to verify 
    behavior when a scope is applied twice in sequence.
    """
    
    # Simulating configuration flags
    custom_scope_name = "joint_custom_pre_pass"
    constant_folding_enabled = True

    # --- Block 1: Original Custom Pre-Pass ---
    if custom_scope_name is not None:
        with tf.keras.name_scope(custom_scope_name):
            # Simulate applying a graph pass
            op_1 = tf.constant(1.0, name="initial_transform")

    # --- Intermediate Pass: remove_noop_ops ---
    # Corresponds to the GraphTransformObserver call for remove_noop_ops
    with tf.name_scope("remove_noop_ops"):
        op_2 = tf.constant(2.0, name="cleaned_op")

    # --- Conditional Pass: constant_fold_uniform_value ---
    if constant_folding_enabled:
        with tf.name_scope("constant_fold_uniform_value"):
            op_3 = tf.constant(3.0, name="folded_op")

    # --- Block 2: Duplicate Custom Pre-Pass (The Merge Mistake) ---
    # This block mimics the duplicate code found in the PyTorch bug report.
    if custom_scope_name is not None:
        with tf.keras.name_scope(custom_scope_name):
            # Simulate applying the graph pass again
            op_4 = tf.constant(4.0, name="repeated_transform")

    # --- Verification ---
    # Verify that operations are created and scoped correctly.
    # In the PyTorch bug, the pass ran twice. Here, we verify the scope 
    # is applied to the operations in both blocks.
    
    assert "joint_custom_pre_pass" in op_1.name, \
        f"Expected scope in {op_1.name}"
    
    assert "remove_noop_ops" in op_2.name, \
        f"Expected scope in {op_2.name}"
    
    assert "constant_fold_uniform_value" in op_3.name, \
        f"Expected scope in {op_3.name}"
    
    assert "joint_custom_pre_pass" in op_4.name, \
        f"Expected scope in {op_4.name} (from duplicate block)"

    print("Test Passed: Duplicate scope logic executed without errors.")
    print(f"Op 1 (First block): {op_1.name}")
    print(f"Op 4 (Second block): {op_4.name}")

if __name__ == "__main__":
    test_duplicate_scope_application()