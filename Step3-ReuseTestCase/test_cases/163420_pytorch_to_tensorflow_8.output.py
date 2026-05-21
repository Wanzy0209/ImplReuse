import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1):
    # Using the requested API: tf.keras.name_scope
    # This context manager groups operations together, similar to how graph nodes are organized.
    with tf.keras.name_scope("fill_diagonal_scope"):
        # PyTorch: t2 = t0.clone()
        # In TensorFlow, tensors are immutable, so operations return new tensors.
        # We proceed directly to the operation that modifies the data.
        
        # PyTorch: t2.fill_diagonal_(t1.item())
        # TensorFlow equivalent: tf.linalg.set_diag
        # arg0 shape: (1, 1), arg1 shape: () (scalar)
        # The diagonal of a (1, 1) matrix has length 1.
        
        # We need to construct the diagonal part from the scalar arg1.
        # tf.fill creates a tensor of a given shape filled with a value.
        # We calculate the minimum dimension to determine diagonal size.
        diag_size = tf.minimum(tf.shape(arg0)[0], tf.shape(arg0)[1])
        diagonal_values = tf.fill([diag_size], arg1)
        
        # Apply the diagonal to the matrix
        t2 = tf.linalg.set_diag(arg0, diagonal_values)
        return t2

# Setup inputs
# PyTorch: arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
# TensorFlow: Use tf.Variable for gradients, initialized with ones for reproducibility
arg0 = tf.Variable(tf.ones([1, 1], dtype=tf.float32))

# PyTorch: arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)
# TensorFlow: 0-d tensor (scalar) variable
arg1 = tf.Variable(5.0, dtype=tf.float32)

if __name__ == '__main__':
    # 1. Eager Execution
    print("Running Eager Mode...")
    with tf.GradientTape() as tape_eager:
        out_eager = foo(arg0, arg1)
        loss_eager = tf.reduce_sum(out_eager)
    
    grad_eager = tape_eager.gradient(loss_eager, [arg0, arg1])
    print(f"Eager Output: {out_eager.numpy()}")
    print(f"Eager Gradients: {[g.numpy() if g is not None else None for g in grad_eager]}")
    print('Eager Success! ')

    # 2. Compiled Execution (tf.function)
    # This mimics the torch.compile behavior in the original bug report.
    # It traces the function and creates a TensorFlow graph.
    print("\nRunning Compiled Mode (tf.function)...")
    compiled_foo = tf.function(foo)
    
    with tf.GradientTape() as tape_compiled:
        out_compiled = compiled_foo(arg0, arg1)
        loss_compiled = tf.reduce_sum(out_compiled)
        
    grad_compiled = tape_compiled.gradient(loss_compiled, [arg0, arg1])
    print(f"Compiled Output: {out_compiled.numpy()}")
    print(f"Compiled Gradients: {[g.numpy() if g is not None else None for g in grad_compiled]}")

    # 3. Verification
    # Check if Eager and Compiled results match
    outputs_match = np.allclose(out_eager.numpy(), out_compiled.numpy())
    grads_match = all(
        np.allclose(g_e.numpy(), g_c.numpy()) if (g_e is not None and g_c is not None) else (g_e is None and g_c is None)
        for g_e, g_c in zip(grad_eager, grad_compiled)
    )

    if outputs_match and grads_match:
        print('Compile Success! ')
    else:
        print('Compile Failure!  - Divergence detected between Eager and Compiled modes')
        raise AssertionError("Divergence detected")