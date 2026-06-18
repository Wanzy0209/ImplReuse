import tensorflow as tf

# The original issue (Issue 166042) reported an eager/compile divergence 
# and an assertion failure ("int" expected vs "bfloat16" found) in PyTorch.
# This test case targets the similar API 'tf.keras.backend.epsilon'.
# We verify that 'epsilon' behaves consistently between eager and compiled modes
# and interacts correctly with bfloat16 tensors, mirroring the context of the bug.

def test_epsilon_bfloat16_consistency():
    # 1. Verify the return type of epsilon (mimicking the assertion logic in the bug report)
    eps = tf.keras.backend.epsilon()
    assert isinstance(eps, float), f"Expected epsilon to be float, got {type(eps)}"

    # 2. Define a computation using epsilon with bfloat16 (matching the bug report's dtype context)
    @tf.function
    def compiled_op(x):
        # Use epsilon in a calculation
        return x + tf.keras.backend.epsilon()

    # 3. Create a bfloat16 tensor
    bf16_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.bfloat16)

    # 4. Execute in Eager mode
    eager_result = bf16_tensor + eps

    # 5. Execute in Compiled mode
    compiled_result = compiled_op(bf16_tensor)

    # 6. Assert consistency (checking for eager/compile divergence)
    # We check if the results are equal to ensure no divergence occurred.
    assert tf.reduce_all(tf.equal(eager_result, compiled_result)), \
        f"Eager/Compile divergence detected! Eager: {eager_result}, Compiled: {compiled_result}"

    print("Test Passed: Epsilon is consistent with bfloat16 in eager and compiled modes.")

if __name__ == "__main__":
    test_epsilon_bfloat16_consistency()