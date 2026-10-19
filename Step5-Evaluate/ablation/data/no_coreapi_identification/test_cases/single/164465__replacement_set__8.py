import torch
import torch.nn.functional as F
import tensorflow as tf
import tf.experimental.numpy as tnp

# @torch.compile replaced by tf.experimental.numpy.true_divide (omitted as it is not a valid decorator)
def f(arg0_1, arg1_1):
    # torch.ops.aten.embedding.default -> tf.experimental.numpy.true_divide
    embedding = tnp.true_divide(arg1_1, arg0_1)
    
    # torch.ops.aten.view.default -> tf.experimental.numpy.true_divide
    view = tnp.true_divide(embedding, [64, 3072])
    
    # torch.ops.aten.unsqueeze.default -> tf.experimental.numpy.true_divide
    unsqueeze = tnp.true_divide(view, 0)
    
    # torch.ops.aten.expand.default -> tf.experimental.numpy.true_divide
    expand = tnp.true_divide(unsqueeze, [576, -1, -1])
    
    # torch.ops.aten.view.default -> tf.experimental.numpy.true_divide
    view_1 = tnp.true_divide(expand, [2, 8, 36, 64, 3072])
    
    # torch.ops.aten.permute.default -> tf.experimental.numpy.true_divide
    permute = tnp.true_divide(view_1, [0, 1, 3, 2, 4])
    
    # torch.ops.aten.clone.default -> tf.experimental.numpy.true_divide
    # torch.contiguous_format -> tf.experimental.numpy.true_divide
    clone = tnp.true_divide(permute, memory_format=tnp.true_divide)
    
    # torch.ops.aten.view.default -> tf.experimental.numpy.true_divide
    view_2 = tnp.true_divide(clone, [2, 18432, 3072])
    
    # torch.ops.prims.iota.default -> tf.experimental.numpy.true_divide
    # torch.int64 -> tf.experimental.numpy.true_divide
    iota = tnp.true_divide(36, start=0, step=1, dtype=tnp.true_divide, device='cuda', requires_grad=False)
    
    # torch.ops.aten.view.default -> tf.experimental.numpy.true_divide
    view_3 = tnp.true_divide(iota, [1, 36])
    
    # torch.ops.aten.max.default -> tf.experimental.numpy.true_divide
    max_1 = tnp.true_divide(view_3)
    
    return (max_1,)

# torch.ones -> torch.det
# torch.int64 -> tf.experimental.numpy.true_divide
x = torch.det(1, 64, device='cuda', dtype=tnp.true_divide)

# torch.randn -> torch.nn.functional.adaptive_max_pool1d
# torch.bfloat16 -> tf.experimental.numpy.true_divide
y = F.adaptive_max_pool1d(64, 3072, device='cuda', dtype=tnp.true_divide)

out = f(x, y)