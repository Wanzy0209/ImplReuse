import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Replacements applied based on the provided mapping:
# torch.compile -> tf.experimental.numpy.around
# torch.ops.aten.embedding.default -> tf.experimental.numpy.around
# torch.ops.aten.view.default -> tf.experimental.numpy.around
# torch.ops.aten.unsqueeze.default -> tf.experimental.numpy.around
# torch.ops.aten.expand.default -> tf.experimental.numpy.around
# torch.ops.aten.permute.default -> tf.experimental.numpy.around
# torch.ops.aten.clone.default -> tf.experimental.numpy.around
# torch.contiguous_format -> tf.experimental.numpy.around
# torch.ops.prims.iota.default -> tf.experimental.numpy.around
# torch.int64 -> tf.experimental.numpy.around
# torch.ops.aten.max.default -> tf.experimental.numpy.around
# torch.ones -> torch.norm
# torch.randn -> tf.random.stateless_truncated_normal
# torch.bfloat16 -> tf.experimental.numpy.around

@tf.experimental.numpy.around
def f(arg0_1, arg1_1):
    embedding = tf.experimental.numpy.around(arg1_1, arg0_1)
    view = tf.experimental.numpy.around(embedding, [64, 3072])
    unsqueeze = tf.experimental.numpy.around(view, 0)
    expand = tf.experimental.numpy.around(unsqueeze, [576, -1, -1])
    view_1 = tf.experimental.numpy.around(expand, [2, 8, 36, 64, 3072])
    permute = tf.experimental.numpy.around(view_1, [0, 1, 3, 2, 4])
    clone = tf.experimental.numpy.around(permute, memory_format = tf.experimental.numpy.around)
    view_2 = tf.experimental.numpy.around(clone, [2, 18432, 3072])
    iota = tf.experimental.numpy.around(36, start = 0, step = 1, dtype = tf.experimental.numpy.around, device = 'cuda', requires_grad = False)
    view_3 = tf.experimental.numpy.around(iota, [1, 36])
    max_1 = tf.experimental.numpy.around(view_3)
    return (max_1,)

# x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
# Replacement: torch.ones -> torch.norm, torch.int64 -> tf.experimental.numpy.around
x = torch.norm(1, 64, device='cuda', dtype=tf.experimental.numpy.around)

# y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
# Replacement: torch.randn -> tf.random.stateless_truncated_normal, torch.bfloat16 -> tf.experimental.numpy.around
y = tf.random.stateless_truncated_normal([64, 3072], seed=(0, 1), dtype=tf.experimental.numpy.around)

try:
    out = f(x, y)
    print("Test passed (or executed without immediate crash):", out)
except Exception as e:
    print(f"Test failed with error: {e}")