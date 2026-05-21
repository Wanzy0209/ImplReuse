import torch
import tensorflow as tf
import numpy as np

def test_tf_strings_as_string():
    """
    Adapted test case for tf.strings.as_string based on the PyTorch bug report.
    The original bug involved torch.tanh on int64 inputs causing a divergence
    between eager and compiled modes. This test verifies the behavior of the
    similar API, tf.strings.as_string, with int64 inputs in both eager and
    compiled (XLA) contexts.
    """
    
    # Setup inputs similar to the original bug report
    # The original bug was triggered by operations on int64 tensors
    arg0 = tf.constant(np.random.randint(0, 1000, [42, 56]), dtype=tf.int64)

    def foo(arg0):
        # Original API: torch.tanh(t0)
        # Similar API: tf.strings.as_string(arg0)
        # We apply the string conversion to the int64 tensor
        t1 = tf.strings.as_string(arg0)
        
        # In the original code, there were subsequent operations (clone, zero_, etc.)
        # Since t1 is now a string tensor, we cannot perform float math directly.
        # We perform a string operation to maintain a non-trivial graph.
        # t2 = t1 (simulating the flow)
        return t1

    # 1. Test Eager Mode
    print("Testing Eager Mode...")
    try:
        out_eager = foo(arg0)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')
        return

    # 2. Test Compiled Mode (tf.function with XLA)
    print("Testing Compiled Mode...")
    compiled_foo = tf.function(foo, jit_compile=True)
    
    try:
        out_compiled = compiled_foo(arg0)
        
        # Verify consistency between Eager and Compiled outputs
        # tf.strings.as_string returns byte strings
        if tf.reduce_all(out_eager == out_compiled):
            print('Compile Success! ')
        else:
            print('Compile Divergence Detected! ')
            print("Eager Output:", out_eager)
            print("Compiled Output:", out_compiled)
            
    except Exception as e:
        print(f'Compile Failed with error: {e}')

if __name__ == '__main__':
    test_tf_strings_as_string()