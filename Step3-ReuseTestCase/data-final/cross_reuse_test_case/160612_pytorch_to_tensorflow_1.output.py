import tensorflow as tf

# Create a dataset
dataset = tf.data.Dataset.range(5)

# Define the scan function (e.g., cumulative sum)
initial_state = tf.constant(0, dtype=tf.int64)
def scan_func(state, element):
    new_state = state + element
    return new_state, new_state

# Apply the scan transformation
scanned_dataset = dataset.apply(tf.data.experimental.scan(initial_state, scan_func))

# Verify the results
expected_results = [0, 1, 3, 6, 10]
actual_results = list(scanned_dataset.as_numpy_iterator())

assert actual_results == expected_results, f"Expected {expected_results}, got {actual_results}"
print("Test passed.")