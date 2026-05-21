import torch
import tensorflow as tf

def foo(arg0, arg1):
    # Use tf.compat.v1.name_scope to organize the operations, 
    # analogous to the scope provided by torch.compile in the original report.
    with tf.compat.v1.name_scope("fill_diagonal_scope"):
        # t2 = t0.clone()
        # In TensorFlow, operations are functional. tf.identity creates a copy of the tensor.
        t2 = tf.identity(arg0)
        
        # t2.fill_diagonal_(t1.item())
        # TensorFlow does not have an in-place fill_diagonal_.
        # We use tf.linalg.set_diag to achieve the same result.
        # arg0 is (1, 1), so the diagonal is a vector of length 1.
        # arg1 is a scalar, so we reshape it to match the diagonal length.
        diagonal_values = tf.reshape(arg1, [1])
        t2 = tf.linalg.set_diag(t2, diagonal_values)
        
        return t2

if __name__ == '__main__':
    # Setup inputs
    # PyTorch: torch.empty([1, 1], dtype=torch.float32, device='cuda')
    # TensorFlow: tf.zeros (or random) on CPU/GPU
    arg0 = tf.zeros([1, 1], dtype=tf.float32)
    
    # PyTorch: torch.empty([], dtype=torch.float32, device='cuda')
    # TensorFlow: scalar constant
    arg1 = tf.constant(5.0, dtype=tf.float32)

    # 1. Eager Execution
    out_eager = foo(arg0, arg1)
    print('Eager Output:', out_eager.numpy())
    
    # 2. Compiled Execution (tf.function is the TF equivalent to torch.compile)
    # We wrap the function to test for eager/compile divergence as in the original bug.
    compiled_foo = tf.function(foo)
    out_compiled = compiled_foo(arg0, arg1)
    print('Compiled Output:', out_compiled.numpy())

    # Verify results match
    if tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy():
        print('TensorFlow Success! ')
    else:
        print('TensorFlow Divergence! ')