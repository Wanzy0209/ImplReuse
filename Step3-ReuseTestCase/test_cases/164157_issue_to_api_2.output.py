import torch
import tensorflow as tf
import numpy as np

def foo(arg3, arg4, arg5):
    """
    Replicates the tensor manipulation logic from the bug report,
    replacing torch.std with the similar API tf.compat.v1.argmax.
    """
    # t6, t7, t8: size=(256, 88, 1), dtype=float16
    t6 = arg3
    t7 = arg4
    t8 = arg5

    # t9 = torch.cat([t6, t6, t7, t8], dim=2)
    # size=(256, 88, 4), dtype=float16
    t9 = tf.concat([t6, t6, t7, t8], axis=2)

    # Original API: t10 = t9.std(dim=2)
    # Similar API: tf.compat.v1.argmax
    # The bug report highlights issues with reduction operations on float16 tensors.
    # We test argmax on the same tensor structure.
    t10 = tf.compat.v1.argmax(t9, axis=2)

    return t10

# Setup inputs matching the bug report's characteristics
# size=(256, 88, 1), dtype=float16
arg3 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)
arg4 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)
arg5 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)

if __name__ == '__main__':
    # Run the function
    out = foo(arg3, arg4, arg5)
    
    # Verify output characteristics
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    
    # Assertions to ensure correct behavior
    assert out.shape == (256, 88), f"Shape mismatch: expected (256, 88), got {out.shape}"
    assert out.dtype == tf.int64, f"Dtype mismatch: expected int64, got {out.dtype}"
    
    print('Test Success! ')