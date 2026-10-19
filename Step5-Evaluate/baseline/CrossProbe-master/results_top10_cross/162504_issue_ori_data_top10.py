```python
import tensorflow as tf

# PyTorch: torch.cuda.manual_seed(42)
# TF: Set global seed. Note: For strict reproducibility in graphs, 
# especially with checkpointing, stateless ops are preferred.
tf.random.set_seed(42)

# PyTorch: def fn(x): ...
# TF: We use @tf.recompute_grad to mimic torch.utils.checkpoint.checkpoint.
# We pass a seed to ensure deterministic behavior during recomputation.
@tf.recompute_grad
def fn(x, seed):
    # PyTorch: torch.randn(1, device="cuda")
    # TF: tf.random.stateless_normal ensures that if the function is recomputed
    # (which happens during the backward pass with recompute_grad),
    # it generates the same random numbers, matching preserve_rng_state=True.
    rand_val = tf.random.stateless_normal((1,), seed=seed)
    return x * tf.math.sigmoid(rand_val)

# PyTorch: initialize device state
# TF: Not strictly required, but kept for structural similarity.
# fn(tf.ones((1,)), seed=tf.constant([42, 42], dtype=tf.int32))

# --- Eager Execution ---
# PyTorch: torch.cuda.manual_seed(42)
# We use a fixed seed tensor for the stateless operation.
seed = tf.constant([42, 42], dtype=tf.int32)

# PyTorch: eager_in = torch.ones(1, device="cuda", requires_grad=True)
eager_in = tf.ones((1,), dtype=tf.float32)

# PyTorch: eager_out = checkpoint(...)
# PyTorch: eager_in_grad = autograd.grad(...)
# We wrap this in a function to easily convert to graph mode later.
def compute_grad(x, seed):
    with tf.GradientTape() as tape:
        # PyTorch: requires_grad=True
        tape.watch(x)
        out = fn(x, seed)
    return tape.gradient(out, x)

eager_in_grad = compute_grad(eager_in, seed)

# --- Graph Capture ---
# PyTorch: g = torch.cuda.CUDAGraph()
# PyTorch: with torch.cuda.graph(g): ...
# TF: We use tf.function to capture the computation graph.
@tf.function
def graph_step(x, seed):
    return compute_grad(x, seed)

# PyTorch: torch.cuda.manual_seed(42)
# TF: Seed is already set in the variable `seed`. 
# For stateless ops, we don't need to reset global seed, 
# just pass the same seed tensor.

# PyTorch: g.replay()
# In TF, calling the tf.function executes the captured graph.
graph_in_grad = graph_step(eager_in, seed)

# PyTorch: assert torch.allclose(...)
# TF: assert
# Note: The original code asserts rtol=0, atol=0.
# This requires exact bit-level match.
# Stateless ops with the same seed guarantee this.
tf.debugging.assert_all_close(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0, message="Mismatch in gradient outputs")

print(eager_in_grad.numpy())
print(graph_in_grad.numpy())
```