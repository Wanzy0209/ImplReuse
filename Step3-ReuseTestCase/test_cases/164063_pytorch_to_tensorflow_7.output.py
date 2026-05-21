import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Enable numpy behavior for tf.experimental.numpy
tnp.experimental_enable_numpy_behavior()

def foo(arg0, arg1):
    # Original bug context: operations on bfloat16 tensors
    # t0 = arg0 # size=(28, 24, 3, 127), dtype=bfloat16
    
    # Original API: t2 = t1.var(dim=2)
    # Adapted API: tf.experimental.numpy.power
    # We test power with bfloat16 inputs to check for type handling issues similar to the original bug.
    # Note: var is a reduction, power is element-wise. We adapt the test to focus on the API behavior.
    t2 = tnp.power(arg0, arg1)
    
    # Add a sentinel operation to ensure the graph is connected and executed
    sentinel = tf.constant(0.0, dtype=tf.bfloat16)
    output = t2 + sentinel
    return output

# Setup inputs similar to the original bug report (bfloat16)
# Original t1 shape was (28, 24, 3, 127)
arg0 = tf.random.normal([28, 24, 3, 127], dtype=tf.bfloat16) 
arg1 = tf.constant(2.0, dtype=tf.bfloat16) # Squaring, similar to the operation in variance

if __name__ == '__main__':
    print("Testing tf.experimental.numpy.power with bfloat16...")
    
    # Eager execution
    print("Running Eager...")
    try:
        out_eager = foo(arg0, arg1)
        print(f"Eager Success! Output shape: {out_eager.shape}, dtype: {out_eager.dtype}")
    except Exception as e:
        print(f"Eager Error: {e}")

    # Compiled execution (mimicking torch.compile)
    print("Running Compiled...")
    try:
        compiled_foo = tf.function(foo, jit_compile=True)
        out_compiled = compiled_foo(arg0, arg1)
        print(f"Compile Success! Output shape: {out_compiled.shape}, dtype: {out_compiled.dtype}")
    except Exception as e:
        print(f"Compile Error: {e}")