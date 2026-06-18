import tensorflow as tf

# Setup from the original bug report
keys = range(10)
allowed = [0, 1, 2, 3]

# Using the Similar API: tf.VariableAggregation
# We define a variable with aggregation to introduce state management complexity
# similar to the context of the original bug.
agg_var = tf.Variable(0.0, aggregation=tf.VariableAggregation.MEAN)

@tf.function
def fn(x):
    x = x + 1
    # The problematic pattern: list comprehension creating a local variable 'key'
    key = [k for k in keys if k in allowed]

    def inner():
        nonlocal key
        # Interaction with the similar API inside the nested scope
        return agg_var.read_value()

    # Execute inner to trigger nonlocal access
    inner()

    return x + key[0]

# Run the test
# This verifies that TensorFlow's compiler (tf.function) handles the
# local/cell variable name conflict correctly, unlike the reported PyTorch bug.
result = fn(tf.ones(3))
assert result.shape == (3,)
print("Test passed.")