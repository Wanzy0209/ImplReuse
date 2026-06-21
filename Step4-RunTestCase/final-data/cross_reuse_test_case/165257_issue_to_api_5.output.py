import tensorflow as tf

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

# Check execution mode to handle both Eager and Graph modes
if tf.executing_eagerly():
    # Eager execution path
    for name, op_func in get_ops():
        var = tf.Variable(tf.zeros([50, 50]))

        with tf.control_dependencies([op_func(var)]):
            result = tf.identity(var)

        max_val = result.numpy().max()
        status = " OK" if max_val != 0.0 else " FAIL"
        
        print(f"{name:<20} {max_val:<12.4f} {status}")
else:
    # Graph execution path (tf.compat.v1.Session)
    with tf.compat.v1.Session() as sess:
        for name, op_func in get_ops():
            var = tf.Variable(tf.zeros([50, 50]))

            with tf.control_dependencies([op_func(var)]):
                result = tf.identity(var)
            
            # Initialize the variable before running
            sess.run(var.initializer)
            
            # Evaluate the result tensor
            val = sess.run(result)
            max_val = val.max()
            status = " OK" if max_val != 0.0 else " FAIL"
            
            print(f"{name:<20} {max_val:<12.4f} {status}")