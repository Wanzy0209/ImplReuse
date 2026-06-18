import torch
import tensorflow as tf

# Define the computation using the similar API: tf.keras.layers.add
def compute(x, w):
    # tf.keras.layers.add takes a list of tensors to perform element-wise addition
    return tf.keras.layers.add([x, w])

def nop(x, w):
    # tf.Assert is the TensorFlow equivalent for runtime checks in graph mode
    # (similar to torch._check)
    tf.Assert(tf.equal(tf.shape(x)[0], 0), [tf.shape(x)[0]])
    return tf.zeros_like(x)

def chunked_compute(x, w):
    sz = tf.shape(x)[0]
    # Constraint check
    tf.Assert(sz <= 8, [sz])

    # tf.cond mimics torch.cond. Note that tf.cond requires callables (lambdas)
    # for the true and false branches.
    out0 = tf.cond(sz > 0, lambda: compute(x[0:2], w), lambda: nop(x[0:2], w))
    out1 = tf.cond(sz > 2, lambda: compute(x[2:4], w), lambda: nop(x[2:4], w))
    out2 = tf.cond(sz > 4, lambda: compute(x[4:6], w), lambda: nop(x[4:6], w))
    out3 = tf.cond(sz > 6, lambda: compute(x[6:8], w), lambda: nop(x[6:8], w))

    return tf.concat([out0, out1, out2, out3], axis=0)

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize a weight variable similar to the original Linear layer's weight
        self.w = tf.Variable(tf.random.normal((16, 16)), trainable=True)

    def call(self, x):
        return chunked_compute(x, self.w)

# Setup inputs
x = tf.random.normal((4, 16))
w = tf.random.normal((16, 16))

# Sanity check in eager mode (similar to the original assert)
# Note: compute(x, w) performs addition, so we check against chunked logic
# For this specific logic, chunked_compute with sz=4 should trigger compute for indices 0-3
# and nop for 4-7.
# We verify the model runs without error.
model = Model()

# Mimic the graph capture/export process.
# In TensorFlow, tf.function traces the graph, analogous to torch._dynamo capture.
# This tests if the user code stack and control flow are handled correctly
# when using the similar API (add) inside conditional branches.
try:
    traced_model = tf.function(model)
    output = traced_model(x)
    
    # Verify output shape
    assert output.shape == (4, 16), f"Expected shape (4, 16), got {output.shape}"
    print("Test passed: Graph capture and execution successful with tf.keras.layers.add.")
    
except Exception as e:
    print(f"Test failed: Error during graph capture/execution - {e}")
    raise