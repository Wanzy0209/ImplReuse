import torch
import tensorflow as tf

def inner(x):
    return x + 1

# Adapted test case using tf.keras.backend.name_scope
# This API provides context (names) to operations, similar to how the bug report
# requests more context (type/value) in the trace logs for torch.compile.
def test_name_scope():
    with tf.keras.backend.name_scope("debug_scope"):
        x = tf.ones(3)
        x = inner(x)
        result = inner(x)

    # Verify that the context (scope name) is applied to the resulting tensor/operation.
    # This checks that the "debug information" (the scope) is correctly attached,
    # analogous to verifying that trace logs contain the expected variable context.
    assert "debug_scope" in result.name, f"Expected 'debug_scope' in {result.name}"
    print(f"Result name: {result.name}")

if __name__ == "__main__":
    test_name_scope()