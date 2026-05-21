import tensorflow as tf
import numpy as np

# Adapt inputs from the original PyTorch bug report to TensorFlow tensors.
# The original bug report used specific shapes and dtypes (bfloat16, float32).
# We preserve these to test the target API against similar data structures.

# arg0: size=(5, 4), dtype=bfloat16
arg0 = tf.random.uniform((5, 4), dtype=tf.bfloat16)
# arg1: size=(5, 1024), dtype=bfloat16
arg1 = tf.random.uniform((5, 1024), dtype=tf.bfloat16)
# arg2: size=(1024, 4), dtype=bfloat16
arg2 = tf.random.uniform((1024, 4), dtype=tf.bfloat16)
# arg3: size=(3, 4, 5, 2), dtype=float32
arg3 = tf.random.uniform((3, 4, 5, 2), dtype=tf.float32)
# arg4: size=(), dtype=float32
arg4 = tf.random.uniform((), dtype=tf.float32)

# Add a complex tensor to verify the positive case of iscomplexobj
arg_complex = tf.complex(tf.random.uniform((2, 2)), tf.random.uniform((2, 2)))

def foo(arg0, arg1, arg2, arg3, arg4, arg_complex):
    # The original test case logic involved numerical operations (addmm, norm, pow).
    # Here we adapt the logic to test the target API: tf.experimental.numpy.iscomplexobj.
    # We check if the inputs from the original bug report are complex (they shouldn't be),
    # and if the explicitly complex tensor is identified correctly.

    t0 = tf.experimental.numpy.iscomplexobj(arg0)
    t1 = tf.experimental.numpy.iscomplexobj(arg1)
    t2 = tf.experimental.numpy.iscomplexobj(arg2)
    t3 = tf.experimental.numpy.iscomplexobj(arg3)
    t4 = tf.experimental.numpy.iscomplexobj(arg4)
    t5 = tf.experimental.numpy.iscomplexobj(arg_complex)

    # Return results for comparison
    return (t0, t1, t2, t3, t4, t5)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg_complex)
    print('Eager Success! ')

    # Compiled execution (Graph mode via tf.function)
    compiled_foo = tf.function(foo)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg_complex)
    print('Compile Success! ')

    # Compare outputs (Eager vs Compiled)
    # iscomplexobj returns a boolean, so we check for exact equality.
    all_match = True
    for i, (e, c) in enumerate(zip(out_eager, out_compiled)):
        if e != c:
            print(f' Output {i} differs: Eager={e}, Compiled={c}')
            all_match = False

    if not all_match:
        print(' Eager and Compiled outputs differ!')
        exit(1)

    # Verify correctness of the API behavior
    # The original inputs (bfloat16, float32) are not complex
    if out_eager[0] or out_eager[1] or out_eager[2] or out_eager[3] or out_eager[4]:
        print(' Incorrectly identified real tensor as complex!')
        exit(1)

    # The complex tensor should be identified as complex
    if not out_eager[5]:
        print(' Incorrectly identified complex tensor as real!')
        exit(1)

    print('Test Passed! ')