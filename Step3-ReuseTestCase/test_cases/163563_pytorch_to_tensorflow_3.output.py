import torch
import tensorflow as tf

# Enable eager execution as requested by the similar API
# This must be called before any other TensorFlow operations
tf.compat.v1.enable_eager_execution()

# Check for GPU availability to match the 'cuda' context of the original bug
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'
print(f"Running on device: {device_name}")

def foo(arg0, arg1, arg2):
    # t0 = arg0
    # torch.sigmoid -> tf.sigmoid
    t1 = tf.sigmoid(arg0)
    
    # t2 = arg1
    t3 = tf.sigmoid(arg1)
    
    # t4 = arg2
    # torch.exp -> tf.exp
    t5 = tf.exp(arg2)
    
    # torch.baddbmm(t1, t3, t5) performs batched matrix multiplication and addition
    # Formula: t1 + (t3 @ t5)
    # t3 shape: (Batch, 6, 256), t5 shape: (Batch, 256, 1) -> result shape: (Batch, 6, 1)
    t6 = t1 + tf.linalg.matmul(t3, t5)
    
    # torch.reshape -> tf.reshape
    t7 = tf.reshape(t6, (193, 386, 459))
    return t7

# Create inputs matching the original shapes and dtype
# Original: torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cuda')
# TF equivalent: tf.random.uniform with dtype=tf.bfloat16
with tf.device(device_name):
    arg0 = tf.random.uniform([5699097, 6, 1], dtype=tf.bfloat16)
    arg1 = tf.random.uniform([5699097, 6, 256], dtype=tf.bfloat16)
    arg2 = tf.random.uniform([5699097, 256, 1], dtype=tf.bfloat16)

if __name__ == '__main__':
    # Test 1: Eager Execution (Enabled by tf.compat.v1.enable_eager_execution)
    try:
        # Mimic the backward pass logic using GradientTape
        with tf.GradientTape() as tape:
            out_eager = foo(arg0, arg1, arg2)
            loss = tf.reduce_sum(out_eager)
        grads = tape.gradient(loss, [arg0, arg1, arg2])
        print('Eager Execution Success! ')
    except tf.errors.ResourceExhaustedError as e:
        print(f'Eager Execution OOM: {e}')
    except Exception as e:
        print(f'Eager Execution Error: {e}')

    # Test 2: Graph/Compiled Mode (tf.function) to check for divergence
    # This mirrors the torch.compile part of the original bug report
    try:
        compiled_foo = tf.function(foo)
        with tf.GradientTape() as tape:
            out_compiled = compiled_foo(arg0, arg1, arg2)
            loss = tf.reduce_sum(out_compiled)
        grads = tape.gradient(loss, [arg0, arg1, arg2])
        print('Graph/Compiled Execution Success! ')
    except tf.errors.ResourceExhaustedError as e:
        print(f'Graph/Compiled Execution OOM: {e}')
    except Exception as e:
        print(f'Graph/Compiled Execution Error: {e}')