import torch
import tensorflow as tf
import numpy as np

def test_heaviside_fp16_divergence():
    """
    Adapted test case for tf.experimental.numpy.heaviside based on the 
    PyTorch std() fuzzer bug (Issue 164157).
    
    The original bug involved an IncompatibleTypeError with fp16 inputs during 
    compilation. This test verifies that tf.experimental.numpy.heaviside 
    handles fp16 inputs correctly without divergence between eager and compiled modes.
    """
    
    # Setup inputs similar to the original bug report
    # Original shapes: (256, 88, 1), dtype=float16
    # We use float16 to stress the type system, similar to the original bug.
    t6 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)
    t7 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)
    sentinel = tf.constant(0.0, dtype=tf.float16)

    # Define the function using the target API: tf.experimental.numpy.heaviside
    # The original bug logic involved a reduction (std), here we test the 
    # similar API (heaviside) with the same data characteristics.
    def foo(x1, x2, sentinel):
        # heaviside(x1, x2) -> 0 if x1<0, x2 if x1==0, 1 if x1>0
        res = tf.experimental.numpy.heaviside(x1, x2)
        # Add sentinel to ensure gradient flow (mimicking original logic)
        return res + sentinel

    # 1. Eager Execution
    with tf.GradientTape() as tape_eager:
        tape_eager.watch(t6)
        tape_eager.watch(t7)
        out_eager = foo(t6, t7, sentinel)
    grad_eager = tape_eager.gradient(out_eager, [t6, t7])
    print('Eager Success! ')

    # 2. Compiled Execution (tf.function equivalent to torch.compile)
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape_compiled:
        tape_compiled.watch(t6)
        tape_compiled.watch(t7)
        out_compiled = compiled_foo(t6, t7, sentinel)
    grad_compiled = tape_compiled.gradient(out_compiled, [t6, t7])
    print('Compile Success! ')

    # 3. Verification (Check for Divergence)
    # The original bug was a divergence/crash. We assert outputs match.
    diff = tf.abs(out_eager - out_compiled)
    # Allow small tolerance for float16 precision
    assert tf.reduce_all(diff < 1e-3).numpy(), f"Output Divergence detected! Max diff: {tf.reduce_max(diff)}"
    
    # Check gradients if they exist (heaviside is 0 almost everywhere)
    if grad_eager[0] is not None and grad_compiled[0] is not None:
        grad_diff = tf.abs(grad_eager[0] - grad_compiled[0])
        assert tf.reduce_all(grad_diff < 1e-3).numpy(), "Gradient Divergence detected!"

    print('Test Passed: No divergence between Eager and Compiled modes.')

if __name__ == '__main__':
    test_heaviside_fp16_divergence()