import tensorflow as tf

def test_local_variables_initializer(mode):
    """
    Test local_variables_initializer behavior in different execution modes.
    This mirrors the structure of the original bug report which tested 
    index_select across different devices (CPU vs MPS).
    """
    # Create a local variable to ensure there is something to initialize
    # Note: In TF2 eager, variables are initialized on creation.
    # We use compat.v1.local_variable to ensure it's tracked as a local variable.
    var = tf.compat.v1.local_variable(initial_value=1.0, name="test_var")

    try:
        if mode == "eager":
            # Eager execution path
            op = tf.compat.v1.local_variables_initializer()
            # In eager mode, this returns a no_op
            print(f"local_variables_initializer test succeeds for mode: {mode}. Op type: {type(op)}")
        elif mode == "graph":
            # Graph execution path (simulated via tf.function)
            @tf.function
            def get_init_op():
                return tf.compat.v1.local_variables_initializer()
            
            op = get_init_op()
            print(f"local_variables_initializer test succeeds for mode: {mode}. Op type: {type(op)}")
    except Exception as e:
        print(f"local_variables_initializer test fails for mode: {mode}: {e}")

# Execute tests for both modes
test_local_variables_initializer(mode="eager")
test_local_variables_initializer(mode="graph")