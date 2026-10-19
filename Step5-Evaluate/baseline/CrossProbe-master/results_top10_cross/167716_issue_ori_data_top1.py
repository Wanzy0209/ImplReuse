```python
import tensorflow as tf

# Conversion: torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse()
# In TensorFlow, sparse tensors are constructed using indices, values, and shape.
# Non-zero elements are at (0,0)=1, (0,2)=2, (1,1)=3.
indices_a = [[0, 0], [0, 2], [1, 1]]
values_a = [1.0, 2.0, 3.0]
shape_a = [2, 3]
# Note: requires_grad_() implies gradient tracking. In TF, this is handled via tf.GradientTape,
# but we define the data structure here.
a = tf.sparse.SparseTensor(indices=indices_a, values=values_a, dense_shape=shape_a)
# Reorder sparse tensor for efficient computation
a = tf.sparse.reorder(a)

# Conversion: torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
# Using tf.Variable to enable gradient tracking equivalent to requires_grad=True
b = tf.Variable([[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]], dtype=tf.float32)

# Conversion: torch.sparse.mm(a, b)
# TensorFlow uses tf.sparse.sparse_dense_matmul for sparse x dense matrix multiplication
y = tf.sparse.sparse_dense_matmul(a, b)

# Conversion: z = y.to_dense()
# tf.sparse.sparse_dense_matmul returns a dense tensor, so z is simply y.
z = y
```