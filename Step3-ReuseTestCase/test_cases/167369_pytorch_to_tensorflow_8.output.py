import torch
import tensorflow as tf

class Config:
    def __repr__(self):
        return "Config()"

# In PyTorch, torch.compile is used to trace the function.
# In TensorFlow, tf.function is the equivalent mechanism for tracing/graph compilation.
# We adapt the test to use tf.keras.name_scope as requested, wrapping the logic inside it.
@tf.function
def forward(x, config):
    # Using tf.keras.name_scope as the similar API context
    with tf.keras.name_scope("repr_trace_test"):
        # Calling repr() on non-constant user object
        # This mimics the core logic of the bug report
        return x * len(repr(config))

if __name__ == "__main__":
    config = Config()
    x = tf.random.normal((2, 2))

    # Execute the traced function
    # This verifies if TensorFlow (via tf.function and name_scope) can handle 
    # tracing repr() on a user-defined object, similar to the PyTorch scenario.
    result = forward(x, config)

    # Verify the result is computed correctly
    # len("Config()") is 8
    expected = x * 8
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Test failed: Output mismatch"
    
    print("Test passed. TensorFlow successfully traced repr() inside name_scope.")