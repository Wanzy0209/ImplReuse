```python
import tensorflow as tf

# Convert PyTorch sparse tensor creation to tf.SparseTensor
# PyTorch: torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse()
# Indices of non-zero elements: (0,0)->1, (0,2)->2, (1,1)->3
indices_a = [[0, 0], [0, 2], [1, 1]]
values_a = [1.0, 2.0, 3.0]
shape_a = [2, 3]
a = tf.SparseTensor(indices=indices_a, values=values_a, dense_shape=shape_a)
# Note: requires_grad_() implies gradient tracking. In TF, use tf.GradientTape.
# SparseTensors are not Variables, so gradients are tracked by watching the values tensor.

# Convert PyTorch tensor with requires_grad=True to tf.Variable
# PyTorch: torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
b = tf.Variable([[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]], dtype=tf.float32)

# Perform sparse-dense matrix multiplication
# PyTorch: y = torch.sparse.mm(a, b)
y = tf.sparse.sparse_dense_matmul(a, b)

# Convert result to dense
# PyTorch: z = y.to_dense()
# Note: tf.sparse.sparse_dense_matmul returns a dense tensor, so this is an identity operation.
z = y
```