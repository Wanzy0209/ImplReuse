import tensorflow as tf

# Helper to normalize results (handles both bool and Tensor returns)
def get_bool_value(val):
    if isinstance(val, tf.Tensor):
        # Use .numpy() to get the concrete value from the tensor
        return bool(val.numpy())
    return bool(val)

# Define check functions separately to avoid lambda complexity and AutoGraph issues
def check_default():
    return tf.compat.v1.executing_eagerly()

@tf.function
def check_inside_function():
    return tf.compat.v1.executing_eagerly()

@tf.function
def check_inside_init_scope():
    # init_scope is a context manager, must use 'with' statement
    with tf.compat.v1.init_scope():
        return tf.compat.v1.executing_eagerly()

contexts = [
    ("Default", check_default),
    ("Inside tf.function", check_inside_function),
    ("Inside init_scope", check_inside_init_scope),
]

print(f"{'Context':<25} {'Expected':<12} {'Actual':<12} {'Status'}")
print("-" * 65)

for name, check_func in contexts:
    # Expected behavior based on TensorFlow documentation
    # Default and init_scope should be eager (True), tf.function should be graph (False)
    expected = True if name in ["Default", "Inside init_scope"] else False
    
    # Execute the check
    result = check_func()
    actual = get_bool_value(result)
    
    # Determine status based on whether the API correctly identifies the context
    status = " OK" if actual == expected else " BUG"
    print(f"{name:<25} {str(expected):<12} {str(actual):<12} {status}")