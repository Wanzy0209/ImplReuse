import torch
import tensorflow as tf
import numpy as np

def foo(images):
    """
    Adapted from PyTorch test case for torch.nn.functional.conv1d.
    Original PyTorch input t4 shape: (17, 64, 358)
    TensorFlow extract_image_patches requires 4D input: [batch, in_rows, in_cols, depth]
    Mapping: (Batch, Channels, Length) -> (Batch, Height=1, Width, Channels)
    Target Shape: (17, 1, 358, 64)
    """
    # PyTorch conv1d parameters: stride=1, padding=0
    # TensorFlow extract_image_patches parameters:
    # ksizes: [1, 1, 1, 1] (kernel size 1x1)
    # strides: [1, 1, 1, 1] (stride 1)
    # rates: [1, 1, 1, 1]
    # padding: 'VALID' (equivalent to padding=0)
    
    patches = tf.compat.v1.extract_image_patches(
        images=images,
        ksizes=[1, 1, 1, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    return patches

# Replicate input characteristics from PyTorch arg1 (t4)
# PyTorch arg1: size=(17, 64, 358), dtype=float32
# Reshaped for TensorFlow: (17, 1, 358, 64)
np_input = np.random.rand(17, 1, 358, 64).astype(np.float32)
images = tf.constant(np_input)

# 1. Run in Eager Mode
print("Running Eager Mode...")
try:
    out_eager = foo(images)
    print(f"Eager Success!  Output shape: {out_eager.shape}")
except Exception as e:
    print(f"Eager Failed!  Error: {e}")

# 2. Run in Compiled Mode (tf.function)
print("\nRunning Compiled Mode...")
try:
    compiled_foo = tf.function(foo)
    out_compiled = compiled_foo(images)
    print(f"Compile Success!  Output shape: {out_compiled.shape}")
except Exception as e:
    print(f"Compile Failed!  Error: {e}")

# 3. Verify Consistency
if 'out_eager' in locals() and 'out_compiled' in locals():
    if tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy():
        print("\nVerification: Eager and Compiled outputs match. ")
    else:
        print("\nVerification: Divergence detected between Eager and Compiled! ")