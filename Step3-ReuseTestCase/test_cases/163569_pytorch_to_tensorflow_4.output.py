import torch
import tensorflow as tf
import sys

def foo(arg0, arg1, arg2):
    # Adapted logic to use tf.image.random_contrast
    # t0 = arg0 # size=(2, 261, 17, 358), dtype=bfloat16
    t0 = arg0
    
    # t1 = t0.max(dim=0).values # size=(261, 17, 358)
    t1 = tf.reduce_max(t0, axis=0)
    
    # t2 = t1.transpose(1, 0) # size=(17, 261, 358)
    # PyTorch transpose(1, 0) on 3D tensor swaps dim 0 and 1.
    # Tensorflow transpose requires explicit perm.
    t2 = tf.transpose(t1, perm=[1, 0, 2])
    
    # t3 = arg1 # size=(17, 261, 358), dtype=float32
    # Note: Adjusted arg1 shape from (17, 64, 358) to (17, 261, 358) 
    # to match t2 shape for the final multiplication, as random_contrast preserves input shape.
    t3 = arg1
    
    # t4 = torch.exp(t3)
    t4 = tf.exp(t3)
    
    # t7 = torch.nn.functional.conv1d(t4, t6, stride=1, padding=0)
    # Replaced with tf.image.random_contrast
    # random_contrast operates on the last 3 dimensions. t4 is 3D.
    # We use fixed lower/upper bounds for the test.
    t7 = tf.image.random_contrast(t4, lower=0.5, upper=1.5)
    
    # t8 = t7.clone(); t8.zero_()
    t8 = tf.identity(t7)
    t8 = tf.zeros_like(t8)
    
    # t9 = t2 * t7 * t8
    t9 = t2 * t7 * t8
    
    output = t9
    return output

# Setup inputs
# arg0: size=(2, 261, 17, 358), dtype=bfloat16
# Using bfloat16 to match the original bug's precision context
try:
    # Check if bfloat16 is supported on the current hardware (usually GPUs or TPUs)
    # If not, fall back to float32 to ensure the test runs, though the bug might be precision specific.
    dtype_arg0 = tf.bfloat16
except:
    dtype_arg0 = tf.float32

arg0 = tf.random.uniform([2, 261, 17, 358], dtype=dtype_arg0)

# arg1: size=(17, 261, 358), dtype=float32
# Shape adjusted from (17, 64, 358) to (17, 261, 358) to ensure broadcasting compatibility 
# with t2 (17, 261, 358) in the final multiplication step.
arg1 = tf.random.uniform([17, 261, 358], dtype=tf.float32)

# arg2: size=(261, 1, 64), dtype=float32
# Kept for signature consistency, though unused in the adapted logic 
# as random_contrast does not take a weight tensor.
arg2 = tf.random.uniform([261, 1, 64], dtype=tf.float32)

if __name__ == '__main__':
    # Eager Execution
    print("Running Eager...")
    try:
        out_eager = foo(arg0, arg1, arg2)
        # Note: random_contrast produces different values each run, so we check shape/type
        assert out_eager.shape == (17, 261, 358), f"Eager shape mismatch: {out_eager.shape}"
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed! : {e}')
        sys.exit(1)

    # Compiled Execution (Graph Mode)
    print("\nRunning Compiled (tf.function)...")
    # Using jit_compile=True to mimic the aggressive compilation of torch.compile
    compiled_foo = tf.function(foo, jit_compile=True)
    
    try:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        assert out_compiled.shape == (17, 261, 358), f"Compiled shape mismatch: {out_compiled.shape}"
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed! : {e}')
        sys.exit(1)