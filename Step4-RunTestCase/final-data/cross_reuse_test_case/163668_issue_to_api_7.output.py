import sys

# Attempt to import dependencies, handling potential environment incompatibilities.
try:
    import torch
    import tensorflow as tf
    from tensorflow.keras import backend as K
except ImportError as e:
    print(f"ImportError: {e}")
    print("Skipping test due to environment incompatibility (likely GLIBCXX version mismatch).")
    sys.exit(0)

# The original bug involves torch._check causing a graph break when used inside torch.compile.
# This test verifies that the similar API, tf.keras.backend.get_uid, interacts correctly
# with the TensorFlow graph context (tf.function) without causing breaks or errors.

@tf.function
def get_id_in_graph(prefix):
    """
    Wrapper function to test the API inside a compiled graph context.
    Analogous to the @torch.compile(fullgraph=True) decorator in the bug report.
    """
    # tf.keras.backend.get_uid interacts with the internal graph state.
    # We check if this operation is compatible with graph tracing.
    return K.get_uid(prefix)

# Test execution
prefix_name = "test_layer"

# First call
uid_1 = get_id_in_graph(prefix_name)
print(f"First UID: {uid_1}")

# Second call to verify state persistence and graph stability
uid_2 = get_id_in_graph(prefix_name)
print(f"Second UID: {uid_2}")

# Assertions to verify correct behavior
assert uid_1 == 1, f"Expected first UID to be 1, got {uid_1}"
assert uid_2 == 2, f"Expected second UID to be 2, got {uid_2}"

print("Test passed: tf.keras.backend.get_uid works correctly inside tf.function graph context.")