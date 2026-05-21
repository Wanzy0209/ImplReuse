import torch
import tensorflow as tf

# Define the function using the similar API (tf.name_scope)
# and the logic from the bug report.
def foo(arg0, arg1):
    with tf.name_scope("fill_diagonal_scope"):
        # Mimic: t2 = t0.clone(); t2.fill_diagonal_(t1.item())
        # In TensorFlow, tensors are immutable. We use tf.linalg.set_diag to mimic the effect.
        # t1 is a 0-d tensor (scalar). tf.linalg.set_diag expects the diagonal to be a vector.
        # For a (1,1) matrix, the diagonal vector has length 1.
        # We reshape t1 to [1] to match the diagonal requirement.
        diagonal_val = tf.reshape(arg1, [1])
        t2 = tf.linalg.set_diag(arg0, diagonal_val)
        return t2

# Inputs
# Mimicking: torch.empty([1, 1], ..., requires_grad=True)
arg0 = tf.Variable(tf.empty([1, 1], dtype=tf.float32))
# Mimicking: torch.empty([], ..., requires_grad=True)
arg1 = tf.Variable(tf.empty([], dtype=tf.float32))

if __name__ == '__main__':
    # Eager Execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1)
        loss_eager = tf.reduce_sum(out_eager)
    grads_eager = tape.gradient(loss_eager, [arg0, arg1])
    print('Eager Success! ')

    # Compiled Execution
    # tf.function is the TensorFlow equivalent to torch.compile
    compiled_foo = tf.function(foo)
    
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1)
        loss_compiled = tf.reduce_sum(out_compiled)
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg1])
    print('Compile Success! ')

    # Verify consistency between Eager and Compiled modes
    assert tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy(), "Output mismatch"
    assert tf.reduce_all(tf.equal(grads_eager[0], grads_compiled[0])).numpy(), "Grad mismatch for arg0"
    assert tf.reduce_all(tf.equal(grads_eager[1], grads_compiled[1])).numpy(), "Grad mismatch for arg1"
    print('Consistency Check Passed! ')