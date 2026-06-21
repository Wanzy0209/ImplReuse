import tensorflow as tf

# Ensure eager execution is enabled to match the script-like nature of the bug report
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Test operations to update the global step
# These mimic the random operations in the original bug report by modifying state
# Fixed: The values being assigned must match the shape of the variable [1, 1]
ops = [
    ("assign(10)", lambda v: v.assign([[10]])),
    ("assign_add(5)", lambda v: v.assign_add([[5]])),
    ("assign_sub(2)", lambda v: v.assign_sub([[2]])),
]

print(f"{'Operation':<20} {'Expected':<12} {'Retrieved':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # Create a variable initialized to 0
    # To reflect the "non-contiguous" aspect of the original bug, we create a variable
    # with a shape [1, 1] and transpose it. While TF eager tensors are values,
    # this setup mirrors the structural complexity of the original bug report.
    var = tf.Variable([[0]], dtype=tf.int64)
    
    # Create a transposed view (analogous to .T in PyTorch)
    # We reshape to scalar before passing to global_step because the API implementation
    # `int(global_step_tensor.numpy())` expects a scalar or single-element tensor.
    var_view = tf.transpose(var)
    var_scalar = tf.reshape(var_view, [])

    # Perform the operation (update the variable)
    op_func(var)

    # Retrieve the step using the similar API: tf.compat.v1.train.global_step
    # This API internally calls `int(global_step_tensor.numpy())`, mirroring the
    # `.item()` call in the original bug report.
    try:
        retrieved_step = tf.compat.v1.train.global_step(None, var_scalar)
        expected = var.numpy()[0][0]
        status = " OK" if retrieved_step == expected else " FAIL"
        print(f"{name:<20} {expected:<12} {retrieved_step:<12} {status}")
    except Exception as e:
        print(f"{name:<20} {'Error':<12} {str(e):<12}  ERROR")