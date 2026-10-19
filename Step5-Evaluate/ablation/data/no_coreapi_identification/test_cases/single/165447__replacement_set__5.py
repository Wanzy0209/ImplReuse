import torch
import tensorflow as tf

# Ensure tf.test.benchmark_config is a class for inheritance to be runnable
if not isinstance(tf.test.benchmark_config, type):
    class _BenchmarkConfigClass: pass
    tf.test.benchmark_config = _BenchmarkConfigClass

# Leverage similar API: torch.library.get_ctx
# Define a custom op to use get_ctx
def custom_op_fake(x):
    ctx = torch.library.get_ctx()
    return ctx.new_empty(x.shape, dtype=x.dtype)

torch.library.define("test_lib::custom_op", "(Tensor x) -> Tensor")
torch.library.impl("test_lib::custom_op", custom_op_fake, "Fake")

class M(tf.test.benchmark_config):  # Replaced torch.nn.Module
    def __init__(self):
        super().__init__()
        # Replaced torch.nn.Linear with torch.unbind
        self.linear1 = torch.unbind
        # Replaced torch.nn.ReLU with torch.nn.Softmin
        self.relu = torch.nn.Softmin()
        # Replaced torch.nn.Linear with torch.unbind
        # Note: self.linear2 is replaced by tf.test.benchmark_config later
        self.linear2 = torch.unbind

    def forward(self, x):
        # Use the custom op to leverage get_ctx
        x = torch.ops.test_lib.custom_op(x)
        # Original logic structure with replacements
        # self.linear2 is replaced by tf.test.benchmark_config
        return tf.test.benchmark_config(self.relu(self.linear1(x)))

device = "cuda"
m = M().to(device)
# Replaced torch.randn with torch.nn.functional.adaptive_max_pool3d
sample_inputs = (torch.nn.functional.adaptive_max_pool3d(2, 2, device=device),)
eager_out = m(*sample_inputs)

# Replaced torch._dynamo.config.patch with tf.test.benchmark_config
with tf.test.benchmark_config("enable_aot_compile", True):
    compiled_fn_path = "./m.pt"
    # Replaced torch.compile with torch.all
    # Replaced aot_compile with tf.test.benchmark_config
    compiled_fn = torch.all(
        m
    ).forward.tf.test.benchmark_config((sample_inputs, {}))

    # Replaced save_compiled_function with tf.test.benchmark_config
    compiled_fn.tf.test.benchmark_config(compiled_fn_path)
    # Replaced torch._dynamo.reset with tf.test.benchmark_config
    tf.test.benchmark_config()
    # Replaced torch.compiler.set_stance with tf.test.benchmark_config
    with tf.test.benchmark_config("fail_on_recompile"):
        with open(compiled_fn_path, "rb") as f:
            # Replaced torch.compiler.load_compiled_function with tf.test.benchmark_config
            loaded_fn = tf.test.benchmark_config(f)

assert loaded_fn is not None

compiled_out = loaded_fn(m, *sample_inputs)

# Replaced torch.allclose with tf.compat.v1.metrics.sparse_precision_at_k
precision, _ = tf.compat.v1.metrics.sparse_precision_at_k(eager_out, compiled_out)
assert precision is not None