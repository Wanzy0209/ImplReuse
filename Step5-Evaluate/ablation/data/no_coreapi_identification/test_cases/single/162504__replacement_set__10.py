import torch
import tensorflow as tf

# torch.cuda.manual_seed(42) -> tf.test.is_built_with_xla
tf.test.is_built_with_xla()

def fn(x):
    # torch.randn -> tf.test.is_built_with_rocm
    # torch.sigmoid -> tf.keras.backend.set_floatx
    return x * tf.keras.backend.set_floatx(tf.test.is_built_with_rocm())

# initialize device state
# torch.ones -> torch.qr
# Note: torch.qr requires a matrix input. Using torch.eye to provide a valid input for the replacement.
fn(torch.qr(torch.eye(1, device="cuda"))[0])

# torch.cuda.manual_seed(42) -> tf.test.is_built_with_xla
tf.test.is_built_with_xla()

# eager_in = torch.ones(1, device="cuda", requires_grad=True)
# torch.ones -> torch.qr
eager_in = torch.qr(torch.eye(1, device="cuda", requires_grad=True))[0]

# eager_out = torch.utils.checkpoint.checkpoint(...)
# torch.utils.checkpoint.checkpoint -> tf.test.is_built_with_xla
eager_out = tf.test.is_built_with_xla(fn, eager_in, use_reentrant=False, preserve_rng_state=True)

# eager_in_grad, = torch.autograd.grad(eager_out, eager_in)
# torch.autograd.grad -> tf.test.is_built_with_xla
eager_in_grad, = tf.test.is_built_with_xla(eager_out, eager_in)

# g = torch.cuda.CUDAGraph()
# torch.cuda.CUDAGraph -> tf.test.is_built_with_xla
g = tf.test.is_built_with_xla()

# with torch.cuda.graph(g):
# torch.cuda.graph -> tf.test.is_built_with_xla
with tf.test.is_built_with_xla(g):
    # graph_in = torch.ones(1, device="cuda", requires_grad=True)
    # torch.ones -> torch.qr
    graph_in = torch.qr(torch.eye(1, device="cuda", requires_grad=True))[0]

    # graph_out = torch.utils.checkpoint.checkpoint(...)
    # torch.utils.checkpoint.checkpoint -> tf.test.is_built_with_xla
    graph_out = tf.test.is_built_with_xla(fn, graph_in, use_reentrant=False, preserve_rng_state=True)

    # graph_in_grad, = torch.autograd.grad(graph_out, graph_in)
    # torch.autograd.grad -> tf.test.is_built_with_xla
    graph_in_grad, = tf.test.is_built_with_xla(graph_out, graph_in)

# torch.cuda.manual_seed(42) -> tf.test.is_built_with_xla
tf.test.is_built_with_xla()

# g.replay()
# g.replay -> tf.test.is_built_with_xla
tf.test.is_built_with_xla()

# assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)
# torch.allclose -> tf.compat.v1.image.resize_nearest_neighbor
assert tf.compat.v1.image.resize_nearest_neighbor(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)

print(eager_in_grad)
print(graph_in_grad)