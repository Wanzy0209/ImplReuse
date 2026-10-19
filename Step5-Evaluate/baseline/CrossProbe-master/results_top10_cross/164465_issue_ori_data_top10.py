```python
import tensorflow as tf

# Conversion: @torch.compile -> @tf.function
@tf.function
def f(arg0_1, arg1_1):
    # Conversion: torch.ops.aten.embedding.default(weight, indices) -> tf.gather(params, indices, axis=0)
    embedding = tf.gather(params=arg1_1, indices=arg0_1, axis=0)
    
    # Conversion: torch.ops.aten.view.default -> tf.reshape
    view = tf.reshape(embedding, [64, 3072])
    
    # Conversion: torch.ops.aten.unsqueeze.default -> tf.expand_dims
    unsqueeze = tf.expand_dims(view, axis=0)
    
    # Conversion: torch.ops.aten.expand.default -> tf.broadcast_to
    expand = tf.broadcast_to(unsqueeze, [576, -1, -1])
    
    # Conversion: torch.ops.aten.view.default -> tf.reshape
    view_1 = tf.reshape(expand, [2, 8, 36, 64, 3072])
    
    # Conversion: torch.ops.aten.permute.default -> tf.transpose
    permute = tf.transpose(view_1, perm=[0, 1, 3, 2, 4])
    
    # Conversion: torch.ops.aten.clone.default -> tf.identity
    clone = tf.identity(permute)
    
    # Conversion: torch.ops.aten.view.default -> tf.reshape
    view_2 = tf.reshape(clone, [2, 18432, 3072])
    
    # Conversion: torch.ops.prims.iota.default -> tf.range
    iota = tf.range(36, dtype=tf.int64)
    
    # Conversion: torch.ops.aten.view.default -> tf.reshape
    view_3 = tf.reshape(iota, [1, 36])
    
    # Conversion: torch.ops.aten.max.default -> tf.reduce_max
    max_1 = tf.reduce_max(view_3)
    
    return (max_1,)

# Conversion: torch.ones -> tf.ones
# Conversion: torch.randn -> tf.random.normal
# Conversion: device='cuda' -> tf.device context
with tf.device('/GPU:0'):
    x = tf.ones((1, 64), dtype=tf.int64)
    y = tf.random.normal((64, 3072), dtype=tf.bfloat16)
    out = f(x, y)
```