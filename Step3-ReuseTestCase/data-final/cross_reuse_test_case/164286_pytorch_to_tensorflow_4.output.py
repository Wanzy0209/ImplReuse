import torch
import tensorflow as tf
import numpy as np

def f(a):
    # In PyTorch, the bug was triggered by a.layout changing from sparse_coo to strided.
    # In TensorFlow, we check if the tensor is a SparseTensor to verify layout preservation.
    is_sparse = isinstance(a, tf.sparse.SparseTensor)
    print(f"Inside f: Is Sparse={is_sparse}, Type={type(a)}")

    # Call the similar API: tf.experimental.numpy.swapaxes
    # Note: swapaxes usually operates on dense tensors, but we test it with sparse here
    # to mimic the original bug's scenario.
    return tf.experimental.numpy.swapaxes(a, 0, 1)

# Create a sparse tensor (analogous to torch.sparse_coo_tensor)
indices = [[0, 1], [1, 2]]
values = [1.0, 2.0]
dense_shape = [3, 3]
a = tf.sparse.SparseTensor(indices, values, dense_shape)

print("--- Direct Call ---")
try:
    result_direct = f(a)
    print(f"Direct call result: {result_direct}")
except Exception as e:
    print(f"Direct call failed: {e}")

print("\n--- Gradient Call (VJP equivalent) ---")
# PyTorch uses torch.func.vjp. TensorFlow uses tf.GradientTape.
try:
    with tf.GradientTape() as tape:
        # We must watch the input tensor to calculate gradients
        tape.watch(a)
        y = f(a)

    # Attempt to compute gradients (analogous to calling the vjp result)
    grads = tape.gradient(y, a)
    print(f"Gradients computed: {grads}")
except Exception as e:
    print(f"Gradient computation failed: {e}")