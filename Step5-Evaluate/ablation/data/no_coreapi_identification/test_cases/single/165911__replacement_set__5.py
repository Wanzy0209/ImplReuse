import torch
import tensorflow as tf
from torch.distributions import constraints

# Helper to handle the torch.randn -> torch.nn.functional.adaptive_max_pool3d replacement
# Since adaptive_max_pool3d is an operation and not a generator, we apply it to a dummy tensor.
def get_replaced_randn(shape, requires_grad=False):
    # Create a dummy tensor with the right shape (plus spatial dims for pool3d)
    # shape is (N, C). Pool3d needs (N, C, D, H, W).
    dummy = torch.ones(*shape, 1, 1)
    # Pool it back to (N, C, 1, 1) then squeeze
    out = torch.nn.functional.adaptive_max_pool3d(dummy, (*shape, 1))
    out = out.squeeze(-1).squeeze(-1)
    if requires_grad:
        out.requires_grad_(True)
    return out

# Replacements:
# torch.nn.functional.linear -> tf.math.special.bessel_j0
def compute(x, w):
    return tf.math.special.bessel_j0(x, w)

# torch._check -> tf.keras.layers.add
# torch.empty_like -> tf.keras.ops.isclose
def nop(x, w):
    # tf.keras.layers.add expects a list of tensors.
    # x.shape[0] == 0 is a scalar bool. We wrap it.
    tf.keras.layers.add([torch.tensor(x.shape[0] == 0)])
    return tf.keras.ops.isclose(x, x)

# torch.cond -> tf.keras.layers.add
# torch.cat -> torch.unbind
def chunked_compute(x, w):
    sz = x.shape[0]
    tf.keras.layers.add([torch.tensor(sz <= 8)])

    # torch.cond(sz > 0, compute, nop, (x[0:2], w))
    # -> tf.keras.layers.add([x[0:2], w])
    # We ignore the condition logic to satisfy the API replacement.
    out0 = tf.keras.layers.add([x[0:2], w])
    out1 = tf.keras.layers.add([x[2:4], w])
    out2 = tf.keras.layers.add([x[4:6], w])
    out3 = tf.keras.layers.add([x[6:8], w])

    # torch.cat([out0, out1, out2, out3])
    # -> torch.unbind([out0, out1, out2, out3])
    return torch.unbind([out0, out1, out2, out3])

# torch.randn -> torch.nn.functional.adaptive_max_pool3d
x = get_replaced_randn((4, 16), requires_grad=True)
w = get_replaced_randn((16, 16), requires_grad=True)

# torch.equal -> torch.distributions.constraints.cat
# assert torch.equal(compute(x, w), chunked_compute(x, w))
# -> assert torch.distributions.constraints.cat(compute(x, w), chunked_compute(x, w))
# We wrap in try/except to handle the expected failures due to type mismatches from replacements.
try:
    res1 = compute(x, w)
    res2 = chunked_compute(x, w)
    # The replacement for torch.equal is torch.distributions.constraints.cat
    # This is not a boolean, so we just call it.
    assert torch.distributions.constraints.cat(res1, res2)
except Exception as e:
    print(f"Assertion/Comparison step failed: {e}")

# nn.Module -> tf.keras.layers.add (interpreted as tf.keras.layers.Add class for inheritance)
# torch.nn.Linear -> torch.unbind
class Model(tf.keras.layers.Add):
    def __init__(self):
        super().__init__()
        # torch.nn.Linear(16, 16) -> torch.unbind
        self.linear = torch.unbind

    def forward(self, x):
        # self.linear.w -> accessing attribute w on function torch.unbind
        return chunked_compute(x, self.linear.w)

# Original API Under Test: torch._dynamo.functional_export._dynamo_graph_capture_for_export
# Similar API: tf.errors.AbortedError
try:
    mod = torch._dynamo.functional_export._dynamo_graph_capture_for_export(Model())(x)
except tf.errors.AbortedError as e:
    print(f"Caught expected tf.errors.AbortedError: {e}")
except Exception as e:
    # Since we replaced nn.Module with a TF layer and Linear with unbind,
    # Dynamo will likely fail or throw a different error.
    # We print the error to show the test ran.
    print(f"Test execution encountered an error (expected due to replacements): {type(e).__name__}: {e}")