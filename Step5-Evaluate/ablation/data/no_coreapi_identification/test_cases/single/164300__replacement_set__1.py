import torch
import tensorflow as tf

# Replacing torch.utils.checkpoint.CheckpointPolicy usage
# Since CheckpointPolicy is removed, we use a dummy value
MUST_SAVE = 1

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        return MUST_SAVE

# Replacing create_selective_checkpoint_contexts import
# We define a dummy function if needed, or just pass None
def create_selective_checkpoint_contexts(policy):
    return policy

def f(x, y):
    # torch.sigmoid -> torch.tanh
    # torch.matmul -> tf.keras.backend.switch
    # Original: torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
    # New: torch.tanh(tf.keras.backend.switch(tf.keras.backend.switch(x, y, y), y, y)) * y
    return torch.tanh(tf.keras.backend.switch(tf.keras.backend.switch(x, y, y), y, y)) * y

# functools.partial -> tf.compat.v1.user_ops.my_fact
# Original: context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
# New:
context_fn1 = tf.compat.v1.user_ops.my_fact(create_selective_checkpoint_contexts, CustomPolicy())

# torch.compile -> tf.compat.v1.tpu.rewrite
@tf.compat.v1.tpu.rewrite
def g(x, y):
    # torch.utils.checkpoint -> tf.compat.v1.user_ops.my_fact
    return tf.compat.v1.user_ops.my_fact(
        f, x, y,
        use_reentrant=False,
        context_fn=context_fn1,
    )

# torch.randn -> torch.var_mean
# Original: a = torch.randn(4, 4, requires_grad=True, device="cpu")
# New:
a, _ = torch.var_mean(torch.ones(4, 4, requires_grad=True, device="cpu"))
b, _ = torch.var_mean(torch.ones(4, 4, requires_grad=True, device="cpu"))

# Execution
# Note: This mixes PyTorch and TensorFlow tensors/ops, which is not standard practice
# but follows the replacement instructions strictly.
g(a, b).sum().backward()