import torch
import tensorflow as tf
import numpy as np

def tf_foo(arg0, arg1, arg2, arg3, arg4, sentinel):
    # t0 = arg0 # size=(36, 7112, 1, 1), dtype=bfloat16
    t0 = arg0
    
    # t1 = t0.reshape((28, 24, 3, 127)) # size=(28, 24, 3, 127), dtype=bfloat16
    t1 = tf.reshape(t0, (28, 24, 3, 127))
    
    # Original: t2 = t1.var(dim=2) # size=(28, 24, 127), dtype=bfloat16
    # Adapted: Use tf.experimental.numpy.isposinf.
    # Note: isposinf is element-wise, so we reduce on axis 2 to match the shape of the original var operation.
    # We also cast back to bfloat16 to maintain type consistency for the subsequent concat operation.
    t2 = tf.cast(tf.reduce_any(tf.experimental.numpy.isposinf(t1), axis=2), tf.bfloat16)
    
    # t3 = arg1 # size=(30, 24), dtype=int64
    t3 = arg1
    
    # t4 = arg2 # size=(512, 127), dtype=bfloat16
    t4 = arg2
    
    # t5 = torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4)
    # TensorFlow equivalent: tf.nn.embedding_lookup
    # Note: embedding_lookup takes (params, ids), while torch.embedding takes (indices, weights)
    t3_clamped = tf.clip_by_value(t3, 0, tf.shape(t4)[0] - 1)
    t5 = tf.nn.embedding_lookup(t4, t3_clamped)
    
    # t6 = arg3 # size=(30, 24, 15), dtype=bfloat16
    t6 = arg3
    
    # t7 = torch.nn.functional.pad(t6, [0, 1], mode='constant', value=0.0)
    # TensorFlow equivalent: tf.pad
    # PyTorch pads the last dimension. TF paddings are [[0,0], [0,0], [0, 1]] for rank 3.
    t7 = tf.pad(t6, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT', constant_values=0.0)
    
    # t8 = arg4 # size=(30, 4, 16, 127), dtype=bfloat16
    t8 = arg4
    
    # t9 = t8.sum(dim=1) # size=(30, 16, 127)
    t9 = tf.reduce_sum(t8, axis=1)
    
    # t10 = torch.baddbmm(t5, t7, t9) -> t5 + bmm(t7, t9)
    # t5: (30, 24, 127), t7: (30, 24, 16), t9: (30, 16, 127)
    # bmm(t7, t9) -> (30, 24, 127)
    bmm_res = tf.linalg.matmul(t7, t9)
    t10 = t5 + bmm_res
    
    # t11 = torch.cat([t2, t10], dim=0) # size=(58, 24, 127)
    t11 = tf.concat([t2, t10], axis=0)
    
    output = t11 + sentinel
    return output

# Setup inputs with bfloat16 to match the original bug context
arg0 = tf.random.uniform((36, 7112, 1, 1), minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
arg1 = tf.random.uniform((30, 24), minval=0, maxval=512, dtype=tf.int64)
arg2 = tf.random.uniform((512, 127), minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
arg3 = tf.random.uniform((30, 24, 15), minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
arg4 = tf.random.uniform((30, 4, 16, 127), minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
sentinel = tf.constant(0.0, dtype=tf.bfloat16)

if __name__ == '__main__':
    # Run Eager
    print("Running Eager...")
    try:
        out_eager = tf_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed! : {e}')

    # Run Compiled (tf.function)
    print("Running Compiled...")
    try:
        compiled_foo = tf.function(tf_foo)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed! : {e}')