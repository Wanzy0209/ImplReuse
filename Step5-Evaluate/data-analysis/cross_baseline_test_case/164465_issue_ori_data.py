```python
import tensorflow as tf

# @tf.function is the TensorFlow equivalent of torch.compile for graph optimization
@tf.function
def f(arg0_1, arg1_1):
    # torch.ops.aten.embedding.default(weight, indices) -> tf.nn.embedding_lookup(params, ids)
    embedding = tf.nn.embedding_lookup(arg1_1, arg0_1)
    
    # torch.ops.aten.view.default -> tf.reshape
    view = tf.reshape(embedding, [64, 3072])
    
    # torch.ops.aten.unsqueeze.default -> tf.expand_dims
    unsqueeze = tf.expand_dims(view, 0)
    
    # torch.ops.aten.expand.default -> tf.broadcast_to
    # PyTorch expand allows -1 to infer size, tf.broadcast_to requires full shape
    expand = tf.broadcast_to(unsqueeze, [576, 64, 3072])
    
    # torch.ops.aten.view.default -> tf.reshape
    view_1 = tf.reshape(expand, [2, 8, 36, 64, 3072])
    
    # torch.ops.aten.permute.default -> tf.transpose
    permute = tf.transpose(view_1, [0, 1, 3, 2, 4])
    
    # torch.ops.aten.clone.default -> tf.identity
    # memory_format is ignored in TF as it handles memory layout automatically
    clone = tf.identity(permute)
    
    # torch.ops.aten.view.default -> tf.reshape
    view_2 = tf.reshape(clone, [2, 18432, 3072])
    
    # torch.ops.prims.iota.default -> tf.range
    # PyTorch iota takes length, TF range takes limit (exclusive)
    iota = tf.range(0, 36, dtype=tf.int64)
    
    # torch.ops.aten.view.default -> tf.reshape
    view_3 = tf.reshape(iota, [1, 36])
    
    # torch.ops.aten.max.default -> tf.reduce_max
    max_1 = tf.reduce_max(view_3)
    
    return (max_1,)


# torch.ones -> tf.ones
# device placement is handled by TensorFlow context or automatically
x = tf.ones((1, 64), dtype=tf.int64)

# torch.randn -> tf.random.normal
y = tf.random.normal((64, 3072), dtype=tf.bfloat16)

out = f(x, y)
```