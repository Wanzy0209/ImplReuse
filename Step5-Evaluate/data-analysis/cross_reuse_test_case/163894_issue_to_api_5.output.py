import tensorflow as tf

def test_tf_inside_function():
    """
    Test case for checking execution context (Eager vs Graph).
    
    Context: This test relates to the PyTorch issue regarding Eager/Compile divergence.
    While the PyTorch bug was a failure in the compiled path, this test verifies
    the TensorFlow API's ability to correctly distinguish between eager execution
    and graph execution (tf.function), which is the fundamental context switch
    involved in the original bug.
    
    Note: tf.inside_function is not a valid TensorFlow API. 
    The correct way to check if code is running inside a tf.function (graph mode)
    is to use 'not tf.executing_eagerly()'.
    """
    
    # 1. Verify behavior in Eager Mode
    # In the original bug, eager mode worked correctly.
    print("Testing Eager Mode...")
    # tf.executing_eagerly() returns True in eager mode, so we negate it to match the logic of "inside_function"
    eager_result = not tf.executing_eagerly()
    assert eager_result is False, "Expected False in eager mode"
    print(f" Eager mode check: {eager_result}")

    # 2. Verify behavior inside tf.function (Graph Mode)
    # In the original bug, the divergence occurred during compilation.
    # We ensure the API correctly identifies the graph context.
    print("Testing Graph Mode (tf.function)...")
    
    @tf.function
    def check_graph_context():
        # tf.executing_eagerly() returns False inside tf.function
        return not tf.executing_eagerly()

    graph_result = check_graph_context()
    assert graph_result is True, "Expected True inside tf.function"
    print(f" Graph mode check: {graph_result}")

    # 3. Demonstrate context-aware branching
    # This mirrors the concept of logic behaving differently in eager vs compile modes.
    def context_aware_op(x):
        if not tf.executing_eagerly():
            # Simulate graph-optimized path
            return x * 2
        else:
            # Simulate eager path
            return x + 1

    val = tf.constant(5)
    
    # Test eager path
    assert context_aware_op(val) == 6
    
    # Test graph path
    assert tf.function(context_aware_op)(val) == 10
    
    print(" Context-aware logic verified.")

if __name__ == "__main__":
    test_tf_inside_function()