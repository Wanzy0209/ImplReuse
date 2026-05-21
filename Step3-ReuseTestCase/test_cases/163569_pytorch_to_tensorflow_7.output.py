import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1):
    # arg0: size=(2, 261, 17, 358), dtype=bfloat16
    # arg1: size=(17, 261, 358), dtype=float32
    # Note: arg1 shape adjusted from (17, 64, 358) to (17, 261, 358) to match 
    # the output dimensionality of the original conv1d operation for the final multiplication.
    
    t0 = arg0
    # PyTorch: t0.max(dim=0).values
    t1 = tf.reduce_max(t0, axis=0) # size=(261, 17, 358)
    
    # PyTorch: t1.transpose(1, 0)
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358)
    
    t3 = arg1
    t4 = tf.exp(t3) # size=(17, 261, 358)
    
    # PyTorch: torch.nn.functional.conv1d(t4, t6, ...)
    # TensorFlow: tf.signal.idct(t4)
    # Replacing conv1d with the similar API idct
    t7 = tf.signal.idct(t4, type=2, norm=None) # size=(17, 261, 358)
    
    # PyTorch: t8 = t7.clone(); t8.zero_()
    t8 = tf.zeros_like(t7) # size=(17, 261, 358)
    
    # PyTorch: t9 = t2 * t7 * t8
    t9 = t2 * t7 * t8 # size=(17, 261, 358)
    
    output = t9
    return output

if __name__ == '__main__':
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    print(f"Running on device: {device}")

    with tf.device(device):
        # Replicating input shapes and dtypes
        # arg0: (2, 261, 17, 358) bfloat16
        arg0 = tf.random.normal([2, 261, 17, 358], dtype=tf.bfloat16)
        
        # arg1: (17, 261, 358) float32
        # Shape adjusted: Original was (17, 64, 358) but conv1d output was (17, 261, 358).
        # To maintain tensor shape compatibility for t2 * t7, we use (17, 261, 358).
        arg1 = tf.random.normal([17, 261, 358], dtype=tf.float32)

    # Test Eager Execution
    print("Testing Eager Execution...")
    try:
        out_eager = foo(arg0, arg1)
        print("Eager Success! ")
    except Exception as e:
        print(f"Eager Failed! : {e}")
        exit(1)

    # Test Compiled Execution (tf.function)
    print("Testing Compiled Execution (tf.function)...")
    try:
        compiled_foo = tf.function(foo)
        out_compiled = compiled_foo(arg0, arg1)
        print("Compile Success! ")
        
        # Verify consistency
        if tf.reduce_all(tf.equal(out_eager, out_compiled)):
            print("Outputs match! ")
        else:
            # Check for numerical closeness if exact equality fails due to precision
            if tf.reduce_all(tf.abs(out_eager - out_compiled) < 1e-5):
                print("Outputs are numerically close! ")
            else:
                print("Outputs differ! ")
                
    except Exception as e:
        print(f"Compile Failed! : {e}")