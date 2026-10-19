```python
import tensorflow as tf

@tf.function
def f(arg0_1, arg1_1):
    # torch.ops.aten.embedding.default(arg1_1, arg0_1)
    # Conversion: PyTorch embedding(weight, indices) -> tf.gather(params, indices, axis=0)
    embedding = tf.gather(arg1_1, arg0_1, axis=0)

    # torch.ops.aten.view.default(embedding, [64, 3072])
    view = tf.reshape(embedding, [64, 3072])

    # torch.ops.aten.unsqueeze.default(view, 0)
    unsqueeze = tf.expand_dims(view, 0)

    # torch.ops.aten.expand.default(unsqueeze, [576, -1, -1])
    expand = tf.broadcast_to(unsqueeze, [576, -1, -1])

    # torch.ops.aten.view.default(expand, [2, 8, 36, 64, 3072])
    view_1 = tf.reshape(expand, [2, 8, 36, 64, 3072])

    # torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4])
    permute = tf.transpose(view_1, perm=[0, 1, 3, 2, 4])

    # torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    # Conversion: TF tensors are immutable, identity creates a copy in the graph
    clone = tf.identity(permute)

    # torch.ops.aten.view.default(clone, [2, 18432, 3072])
    view_2 = tf.reshape(clone, [2, 18432, 3072])

    # torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, device = 'cuda', requires_grad = False)
    # Conversion: iota creates a sequence of integers -> tf.range
    iota = tf.range(36, dtype=tf.int64)

    # torch.ops.aten.view.default(iota, [1, 36])
    view_3 = tf.reshape(iota, [1, 36])

    # torch.ops.aten.max.default(view_3)
    # Conversion: Reduce max over all dimensions
    max_1 = tf.reduce_max(view_3)

    return (max_1,)


# x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
x = tf.ones((1, 64), dtype=tf.int64)

# y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
y = tf.random.normal((64, 3072), dtype=tf.bfloat16)

out = f(x, y)
```