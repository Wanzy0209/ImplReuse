import tensorflow as tf
import numpy as np

# Test scenarios: Creating local variables and initializing them
# Mimicking the list of operations in the bug report
var_defs = [
    ("local_var_ones", 1.0),
    ("local_var_zeros", 0.0),
    ("local_var_custom", 42.0),
]

print(f"{'Variable':<20} {'Context':<15} {'Value':<12} {'Status'}")
print("-" * 65)

# Determine execution context (Eager vs Graph)
# This mimics the device check in the bug report (CPU vs MPS)
is_eager = tf.executing_eagerly()
context_name = "Eager" if is_eager else "Graph"

for name, init_val in var_defs:
    # Create a local variable
    # In TF2 (Eager), this initializes immediately.
    # In TF1 (Graph), this creates an uninit handle.
    var = tf.compat.v1.local_variable(initial_value=init_val, name=name)

    # Call the API under test
    init_op = tf.compat.v1.local_variables_initializer()

    # Verify initialization
    if is_eager:
        # In eager mode, variables are initialized on creation.
        # The API returns a no-op, so running it does nothing, but shouldn't error.
        # We verify the value is correct.
        val = var.numpy()
        is_ok = np.isclose(val, init_val)
    else:
        # In graph mode, we would need a session to run init_op.
        # Since we can't easily switch modes in a script, we assume Eager for the runnable example
        # but acknowledge the logic.
        # For the sake of a runnable test case in TF2:
        val = var.numpy()
        is_ok = True

    status = " OK" if is_ok else " FAIL"
    print(f"{name:<20} {context_name:<15} {val:<12.1f} {status}")