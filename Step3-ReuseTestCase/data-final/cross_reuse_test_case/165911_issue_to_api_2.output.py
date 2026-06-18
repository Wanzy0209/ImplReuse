import torch
import tensorflow as tf

# Enable v2 behavior while allowing compat.v1 usage
tf.compat.v1.enable_v2_behavior()

def compute(x, w):
    # Leverage the similar API: tf.compat.v1.math.add
    # This replaces torch.nn.functional.linear from the original issue
    return tf.compat.v1.math.add(x, w)

def nop(x, w):
    # In the original, torch._check is used. 
    # We use tf.debugging.assert_equal for semantic translation.
    tf.debugging.assert_equal(tf.shape(x)[0], 0)
    return tf.zeros_like(x)

def chunked_compute(x, w):
    sz = tf.shape(x)[0]
    # Translate torch._check to tf.debugging.assert_less_equal
    tf.debugging.assert_less_equal(sz, 8)
    
    # Translate torch.cond to tf.cond
    # Note: tf.cond expects callables with no arguments, so we wrap args in lambdas
    out0 = tf.cond(sz > 0, lambda: compute(x[0:2], w), lambda: nop(x[0:2], w))
    out1 = tf.cond(sz > 2, lambda: compute(x[2:4], w), lambda: nop(x[2:4], w))
    out2 = tf.cond(sz > 4, lambda: compute(x[4:6], w), lambda: nop(x[4:6], w))
    out3 = tf.cond(sz > 6, lambda: compute(x[6:8], w), lambda: nop(x[6:8], w))
    
    return tf.concat([out0, out1, out2, out3], axis=0)

class Model(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.normal((16, 16)), name='w')

    @tf.function
    def __call__(self, x):
        return chunked_compute(x, self.w)

# Test execution
if __name__ == "__main__":
    # Setup inputs similar to the original issue
    x = tf.random.normal((4, 16))
    w = tf.random.normal((16, 16))
    
    # Instantiate model
    model = Model()
    model.w.assign(w) # Initialize with the same w for consistency if needed
    
    # Run the model. The @tf.function decorator triggers graph capture,
    # analogous to torch._dynamo.functional_export._dynamo_graph_capture_for_export
    mod = model(x)
    
    # Basic assertion to verify execution
    assert mod.shape == (4, 16)
    print("Test passed.")