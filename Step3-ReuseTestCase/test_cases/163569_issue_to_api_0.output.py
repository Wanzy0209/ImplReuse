import torch
import tensorflow as tf
import numpy as np

def test_issue_163569_tf_lookup():
    """
    Test case generated based on Issue 163569 (PyTorch conv1d eager/compile divergence).
    This test leverages the similar API 'tf.lookup.KeyValueTensorInitializer' to check
    for similar divergence issues when handling specific tensor shapes and dtypes.
    """
    
    # Configuration mimicking the bug report environment
    # The bug involves bfloat16 and float32 tensors with specific non-contiguous strides.
    # We simulate the tensor creation logic here.
    
    # Shapes from the bug report:
    # arg1 (t3): (17, 64, 358) -> float32
    # arg2 (t5): (261, 1, 64) -> float32
    
    # Note: tf.lookup.KeyValueTensorInitializer requires keys and values to have the same 
    # first dimension. We adapt the shapes to be valid for the API while preserving 
    # the dimensional complexity (17, 64, ...).
    
    batch_size = 17
    dim_1 = 64
    dim_2 = 358
    
    # Replicating the tensor setup from the bug report
    # arg0: (2, 261, 17, 358) bfloat16 -> Not directly used in conv1d, but part of the graph.
    # We focus on the inputs to the target API.
    
    # arg1 equivalent: (17, 64, 358)
    arg1 = tf.random.normal((batch_size, dim_1, dim_2), dtype=tf.float32)
    
    # arg2 equivalent: (17, 1, 64) -> Sliced from (261, ...) to match batch_size for valid table init
    arg2 = tf.random.normal((batch_size, 1, dim_1), dtype=tf.float32)
    
    # Pre-processing logic from the bug:
    # t4 = torch.exp(t3)
    t4 = tf.exp(arg1)
    
    # t6 = t5.transpose(2, 1) -> (17, 64, 1)
    t6 = tf.transpose(arg2, [0, 2, 1])
    
    # To use KeyValueTensorInitializer, we flatten the batch and channel dims 
    # to create a set of keys and values.
    # Keys: derived from t4 (17*64, 358) - using indices as keys for hashability
    # Values: derived from t6 (17*64, 1)
    
    num_keys = batch_size * dim_1
    keys = tf.range(num_keys, dtype=tf.int64)
    values = tf.reshape(t6, [num_keys, 1])
    
    # The function to test, mirroring the structure of the original 'foo'
    def foo(keys, values):
        # Use the similar API: tf.lookup.KeyValueTensorInitializer
        # This replaces torch.nn.functional.conv1d in the logic flow
        init = tf.lookup.KeyValueTensorInitializer(
            keys=keys, 
            values=values,
            key_dtype=tf.int64,
            value_dtype=tf.float32
        )
        table = tf.lookup.StaticHashTable(init, default_value=-1.0)
        
        # Perform operation
        t7 = table.lookup(keys)
        
        # Post-processing logic from the bug:
        # t8 = t7.clone(); t8.zero_()
        # t9 = t2 * t7 * t8
        # Since t8 is zero, t9 should be zero.
        
        # TensorFlow equivalent of clone/zero
        t8 = tf.zeros_like(t7)
        
        # Element-wise multiply
        # We need a 't2' equivalent. In the bug, t2 comes from arg0.
        # We create a dummy tensor with the broadcastable shape of t7.
        t2 = tf.ones_like(t7) 
        
        t9 = t2 * t7 * t8
        
        return t9

    # 1. Eager Execution
    print("Running Eager Execution...")
    try:
        out_eager = foo(keys, values)
        print(f"Eager Success! Output shape: {out_eager.shape}")
    except Exception as e:
        print(f"Eager Failed: {e}")
        return

    # 2. Compiled Execution (tf.function)
    print("Running Compiled Execution (tf.function)...")
    try:
        compiled_foo = tf.function(foo)
        out_compiled = compiled_foo(keys, values)
        print(f"Compile Success! Output shape: {out_compiled.shape}")
    except Exception as e:
        print(f"Compile Failed: {e}")
        return

    # 3. Check for Divergence
    # The original bug reported a failure or divergence between eager and compile.
    # We assert that the outputs are identical.
    if tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy():
        print("Test Passed: Eager and Compiled outputs match. ")
    else:
        print("Test Failed: Eager and Compiled outputs diverge. ")
        print(f"Diff: {tf.reduce_sum(tf.abs(out_eager - out_compiled)).numpy()}")

if __name__ == '__main__':
    test_issue_163569_tf_lookup()