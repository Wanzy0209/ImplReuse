import torch
import tensorflow as tf
import numpy as np

class TestModel(tf.Module):
    def __call__(self, x):
        # Use the similar API: tf.keras.name_scope
        with tf.keras.name_scope("sparse_conversion"):
            # PyTorch: x.to_sparse()
            # TensorFlow: tf.sparse.from_dense
            x_sparse = tf.sparse.from_dense(x)
            
            # PyTorch: x_sparse * 2
            # TensorFlow: SparseTensor supports scalar multiplication
            result = x_sparse * 2
            
            # PyTorch: result.to_dense()
            # TensorFlow: tf.sparse.to_dense
            return tf.sparse.to_dense(result)

# Setup input data
x = tf.random.normal((10, 10))

model = TestModel()

# 1. Test in Eager mode (default in TF2)
print("Eager output:", model(x))

# 2. Test in Compiled mode (using tf.function, which is analogous to torch.compile)
# This traces the function and creates a graph, similar to how torch.compile works.
compiled_model = tf.function(model)
print("Compiled output:", compiled_model(x))

# Verify that the outputs are consistent
# Note: Due to potential graph optimizations, we check if values are close
assert tf.reduce_all(tf.abs(model(x) - compiled_model(x)) < 1e-6).numpy()