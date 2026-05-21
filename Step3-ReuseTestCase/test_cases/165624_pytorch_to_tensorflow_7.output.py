import tensorflow as tf

def test_name_scope_duplicate_application():
    """
    Test case adapted from PyTorch Issue 165624.
    
    The original bug involved a merge mistake where `joint_custom_pre_pass` was 
    executed twice due to duplicate code blocks in `joint_graph.py`.
    
    This test adapts that logic to `tf.keras.backend.name_scope`. It verifies
    the API's behavior when the scoping logic is duplicated (simulating the merge
    mistake). We expect the scope to be applied twice, resulting in nested names.
    """
    
    # Mimic the configuration flag `config.joint_custom_pre_pass`
    use_custom_scope = True
    
    # List to store the names of created variables for verification
    created_names = []

    # --- First Block (Original Code) ---
    if use_custom_scope:
        with tf.keras.backend.name_scope("joint_custom_pre_pass"):
            v1 = tf.Variable(1.0, name="var")
            created_names.append(v1.name)

    # --- Intermediate Block (mimicking remove_noop_ops) ---
    # In the original bug, this operation sits between the two duplicate blocks.
    with tf.keras.backend.name_scope("remove_noop_ops"):
        v2 = tf.Variable(2.0, name="var")
        created_names.append(v2.name)

    # --- Second Block (The Merge Mistake) ---
    # This block was duplicated in the PyTorch source code.
    if use_custom_scope:
        with tf.keras.backend.name_scope("joint_custom_pre_pass"):
            v3 = tf.Variable(3.0, name="var")
            created_names.append(v3.name)

    # --- Assertions ---
    
    # 1. Verify the first variable is inside the first scope instance
    assert "joint_custom_pre_pass/var" in created_names[0], \
        f"Expected 'joint_custom_pre_pass/var', got {created_names[0]}"

    # 2. Verify the intermediate variable is in its own scope
    assert "remove_noop_ops/var" in created_names[1], \
        f"Expected 'remove_noop_ops/var', got {created_names[1]}"

    # 3. Verify the third variable is inside the nested scope.
    # Because the block was duplicated, the scope is entered again, creating a nested path.
    # This confirms the execution flow matches the "buggy" structure.
    assert "joint_custom_pre_pass/joint_custom_pre_pass/var" in created_names[2], \
        f"Expected nested scope 'joint_custom_pre_pass/joint_custom_pre_pass/var', got {created_names[2]}"

    print("Test passed: Duplicate scoping logic results in nested names as expected.")

if __name__ == "__main__":
    test_name_scope_duplicate_application()