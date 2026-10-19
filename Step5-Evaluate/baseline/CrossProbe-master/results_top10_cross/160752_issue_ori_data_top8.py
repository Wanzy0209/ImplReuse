```python
import tensorflow as tf

MAX = 3
BATCH = 37


def func(x, idxs):
    # Conversion: torch.nn.functional.one_hot -> tf.one_hot
    # Conversion: x.square() -> tf.square(x)
    return tf.square(x) * tf.one_hot(idxs, MAX, dtype=x.dtype)


def jacfunc(x, idxs):
    # Conversion: torch.func.jacfwd -> Custom Jacobian using tf.GradientTape
    # TensorFlow does not have a direct forward-mode AD equivalent for full Jacobians 
    # in the core API. We compute the Jacobian by iterating over output dimensions
    # using reverse-mode AD (GradientTape) and vectorization.
    
    # Flatten inputs to handle the Jacobian as a matrix computation
    flat_x = tf.reshape(x, [BATCH * MAX])
    
    # We need to compute the gradient of each output component w.r.t x
    # Output shape is (BATCH, MAX), so there are BATCH * MAX components
    num_outputs = BATCH * MAX
    indices = tf.range(num_outputs)

    def grad_for_output_idx(i):
        with tf.GradientTape() as tape:
            tape.watch(flat_x)
            # Reconstruct x for the function call
            x_reshaped = tf.reshape(flat_x, (BATCH, MAX))
            # Compute function output
            y = func(x_reshaped, idxs)
            # Flatten output to extract the specific component
            flat_y = tf.reshape(y, [num_outputs])
            # Extract the scalar output at index i
            out = flat_y[i]
        
        # Compute gradient of this output w.r.t input
        grad = tape.gradient(out, flat_x)
        return grad

    # Vectorize the gradient computation over all output indices
    # Result shape: (BATCH * MAX, BATCH * MAX)
    jacobian_flat = tf.vectorized_map(grad_for_output_idx, indices)
    
    # Reshape back to (BATCH, MAX, BATCH, MAX) to match PyTorch jacfwd output convention
    # (Output dimensions followed by Input dimensions)
    return tf.reshape(jacobian_flat, (BATCH, MAX, BATCH, MAX))


# Conversion: torch.randint -> tf.random.uniform with int64 cast
idxs = tf.cast(tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int32), tf.int64)
# Conversion: torch.rand -> tf.random.uniform
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```