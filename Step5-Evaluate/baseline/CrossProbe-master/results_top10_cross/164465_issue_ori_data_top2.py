```python
import tensorflow as tf

@tf.function
def f(arg0_1, arg1_1):
    # Convert torch.ops.aten.embedding.default to tf.gather
    # PyTorch embedding takes (weight, indices), tf.gather takes (params, indices, axis)
    embedding = tf.gather(arg1_1, arg0_1, axis=0)
    
    # Convert torch.ops.aten.view.default to tf.reshape
    view = tf.reshape(embedding, [64, 3072])
    
    # Convert torch.ops.aten.unsqueeze.default to tf.expand_dims
    unsqueeze = tf.expand_dims(view, axis=0)
    
    # Convert torch.ops.aten.expand.default to tf.broadcast_to
    expand = tf.broadcast_to(unsqueeze, [576, -1, -1])
    
    # Convert torch.ops.aten.view.default to tf.reshape
    view_1 = tf.reshape(expand, [2, 8, 36, 64, 3072])
    
    # Convert torch.ops.aten.permute.default to tf.transpose
    permute = tf.transpose(view_1, perm=[0, 1, 3, 2, 4])
    
    # Convert torch.ops.aten.clone.default to tf.identity
    # memory_format is implicitly handled in TensorFlow
    clone = tf.identity(permute)
    
    # Convert torch.ops.aten.view.default to tf.reshape
    view_2 = tf.reshape(clone, [2, 18432, 3072])
    
    # Convert torch.ops.prims.iota.default to tf.range
    iota = tf.range(36, dtype=tf.int64)
    
    # Convert torch.ops.aten.view.default to tf.reshape
    view_3 = tf.reshape(iota, [1, 36])
    
    # Convert torch.ops.aten.max.default to tf.reduce_max
    max_1 = tf.reduce_max(view_3)
    
    return (max_1,)


# Convert torch.ones to tf.ones
x = tf.ones((1, 64), dtype=tf.int64)
# Convert torch.randn to tf.random.normal
y = tf.random.normal((64, 3072), dtype=tf.bfloat16)

out = f(x, y)
```