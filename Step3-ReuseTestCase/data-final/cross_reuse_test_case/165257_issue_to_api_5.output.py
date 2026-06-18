import tensorflow as tf

# Test case adapted from PyTorch MPS issue (Issue 165257) to test tf.control_dependencies.
# The original issue tests in-place random operations on non-contiguous tensors.
# Here, we adapt the logic to test that control dependencies ensure execution
# of stateful random operations on variables.

def get_ops():
    """
    Returns a list of operations similar to the PyTorch issue.
    Since TF uses immutable tensors, we use tf.Variable to mimic state mutation.
    """
    return [
        ("normal(0,1)", lambda v: v.assign(tf.random.normal(v.shape))),
        ("uniform(0,1)", lambda v: v.assign(tf.random.uniform(v.shape, maxval=1.0))),
        ("exponential(1)", lambda v: v.assign(tf.random.exponential(v.shape, 1.0))),
        ("random()", lambda v: v.assign(tf.random.uniform(v.shape))),
    ]

print(f"{'Operation':<20} {'Max Value':<12} {'Status'}")
print("-" * 60)

for name, op_func in get_ops():
    # Initialize a variable with zeros (mimicking the tensor in the original bug)
    # We use a Variable to allow in-place-like modification
    var = tf.Variable(tf.zeros([50, 50]))

    # Use tf.control_dependencies to ensure the random operation executes
    # before we read the value. This mirrors the intent of checking if the
    # operation "silently failed" (didn't run) or succeeded.
    with tf.control_dependencies([op_func(var)]):
        # Force execution of the dependency graph
        result = tf.identity(var)

    # Check if the operation actually changed the values (mimicking the bug check)
    # In the original bug, MPS would return 0.0 (fail). Here we expect non-zero.
    max_val = result.numpy().max()
    status = " OK" if max_val != 0.0 else " FAIL"
    
    print(f"{name:<20} {max_val:<12.4f} {status}")