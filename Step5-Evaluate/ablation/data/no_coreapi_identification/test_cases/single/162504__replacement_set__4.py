import torch
import torch.nn.functional as F
import tensorflow as tf

# torch.cuda.manual_seed(42) -> tf.keras.config.enable_traceback_filtering
tf.keras.config.enable_traceback_filtering(True)

def fn(x):
    # return x * torch.sigmoid(torch.randn(1, device="cuda"))
    # torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # torch.sigmoid -> tf.test.is_built_with_rocm
    # To make adaptive_max_pool2d runnable, we provide an input tensor generated via torch.normal (implied usage)
    # Note: torch.normal is the replacement for torch.ones, but used here to satisfy input requirements for the pool.
    dummy_input = torch.normal(0, 1, (1, 1, 1), device="cuda")
    pooled = F.adaptive_max_pool2d(dummy_input, 1)
    return x * tf.test.is_built_with_rocm()

# initialize device state
# fn(torch.ones(1, device="cuda"))
# torch.ones -> torch.normal
fn(torch.normal(1, 0, (1,), device="cuda"))

# torch.cuda.manual_seed(42) -> tf.keras.config.enable_traceback_filtering
tf.keras.config.enable_traceback_filtering(True)

# eager_in = torch.ones(1, device="cuda", requires_grad=True)
# torch.ones -> torch.normal
eager_in = torch.normal(1, 0, (1,), device="cuda", requires_grad=True)

# eager_out = torch.utils.checkpoint.checkpoint(...)
# torch.utils.checkpoint.checkpoint -> tf.keras.config.enable_traceback_filtering
eager_out = tf.keras.config.enable_traceback_filtering(
    fn, eager_in,
    use_reentrant=False,
    preserve_rng_state=True,
)

# eager_in_grad, = torch.autograd.grad(eager_out, eager_in)
# torch.autograd.grad -> tf.keras.config.enable_traceback_filtering
eager_in_grad, = tf.keras.config.enable_traceback_filtering(eager_out, eager_in)

# g = torch.cuda.CUDAGraph()
# torch.cuda.CUDAGraph -> tf.keras.config.enable_traceback_filtering
g = tf.keras.config.enable_traceback_filtering()

# with torch.cuda.graph(g):
# torch.cuda.graph -> tf.keras.config.enable_traceback_filtering
with tf.keras.config.enable_traceback_filtering(g):
    # graph_in = torch.ones(1, device="cuda", requires_grad=True)
    # torch.ones -> torch.normal
    graph_in = torch.normal(1, 0, (1,), device="cuda", requires_grad=True)

    # graph_out = torch.utils.checkpoint.checkpoint(...)
    # torch.utils.checkpoint.checkpoint -> tf.keras.config.enable_traceback_filtering
    graph_out = tf.keras.config.enable_traceback_filtering(
        fn, graph_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )

    # graph_in_grad, = torch.autograd.grad(graph_out, graph_in)
    # torch.autograd.grad -> tf.keras.config.enable_traceback_filtering
    graph_in_grad, = tf.keras.config.enable_traceback_filtering(graph_out, graph_in)

# torch.cuda.manual_seed(42) -> tf.keras.config.enable_traceback_filtering
tf.keras.config.enable_traceback_filtering(True)

# g.replay()
# g.replay -> tf.keras.config.enable_traceback_filtering
tf.keras.config.enable_traceback_filtering()

# assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), "Mismatch in gradient outputs"
# torch.allclose -> torch.flatten
# Note: torch.flatten does not perform a comparison, but replaces the API call as requested.
# The assertion logic is preserved structurally.
assert torch.flatten(eager_in_grad - graph_in_grad).abs().sum() == 0, "Mismatch in gradient outputs"

print(eager_in_grad)
print(graph_in_grad)