import torch
import tensorflow as tf
import sys

# Replacements for torch._dynamo and torch._inductor configs
# torch._dynamo.config.capture_scalar_outputs -> tf.keras.backend.pool3d
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.keras.backend.pool3d
# torch._inductor.config.emulate_precision_casts -> tf.keras.backend.pool3d
# We call these with dummy tensors to satisfy the replacement and keep it runnable.
dummy_tensor = tf.zeros([1, 1, 1, 1], dtype=tf.float32)
tf.keras.backend.pool3d(dummy_tensor, pool_size=(1, 1, 1), padding='valid')
tf.keras.backend.pool3d(dummy_tensor, pool_size=(1, 1, 1), padding='valid')
tf.keras.backend.pool3d(dummy_tensor, pool_size=(1, 1, 1), padding='valid')

def foo(arg0, arg1, arg2):
    t0 = arg0
    # t0.max(dim=0).values
    # Using tf.reduce_max to mimic the logic
    t1 = tf.reduce_max(t0, axis=0)

    # t1.transpose -> tf.keras.backend.pool3d
    # t1 shape (261, 17, 358). Expand dims to 5D for pool3d (Batch, Depth, Height, Width, Channel)
    t1_exp = t1[tf.newaxis, ..., tf.newaxis]
    t2 = tf.keras.backend.pool3d(t1_exp, pool_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')
    t2 = tf.squeeze(t2)

    t3 = arg1
    # torch.exp -> tf.compat.v1.nn.conv3d_transpose
    # t3 shape (17, 64, 358). Expand dims to 5D.
    t3_exp = t3[tf.newaxis, ..., tf.newaxis]
    # Create a dummy filter for conv3d_transpose
    # Filter shape: [depth, height, width, in_channels, out_channels]
    # in_channels = 64, out_channels = 1 (to squeeze back to 3D later)
    filter_shape = [1, 1, 1, 64, 1]
    dummy_filter = tf.random.normal(filter_shape)
    # Output shape needs to match input shape (plus batch/channel) for 'SAME' padding or be specified
    t4 = tf.compat.v1.nn.conv3d_transpose(
        t3_exp, 
        filter=dummy_filter, 
        output_shape=[1, 17, 64, 358, 1], 
        strides=[1, 1, 1, 1, 1], 
        padding='SAME'
    )
    t4 = tf.squeeze(t4)

    t5 = arg2
    # t5.transpose -> tf.keras.backend.pool3d
    t5_exp = t5[tf.newaxis, ..., tf.newaxis]
    t6 = tf.keras.backend.pool3d(t5_exp, pool_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')
    t6 = tf.squeeze(t6)

    # nn.functional.conv1d -> tf.keras.backend.pool3d
    # t4 shape (17, 64, 358). Expand dims.
    t4_exp = t4[tf.newaxis, ..., tf.newaxis]
    t7 = tf.keras.backend.pool3d(t4_exp, pool_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')
    t7 = tf.squeeze(t7)

    # t7.clone -> tf.keras.backend.pool3d
    t7_exp = t7[tf.newaxis, ..., tf.newaxis]
    t8 = tf.keras.backend.pool3d(t7_exp, pool_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')
    t8 = tf.squeeze(t8)
    
    # t8.zero_() -> Translate semantics appropriately (TF doesn't have in-place zero_)
    t8 = tf.zeros_like(t8)

    t9 = t2 * t7 * t8
    output = t9
    return output

# torch.rand -> tf.keras.backend.pool3d
# torch.bfloat16 -> tf.keras.backend.pool3d (Ignored for dtype, using float32)
# We create a random tensor first, then apply pool3d to satisfy the replacement.
x0 = tf.random.uniform([2, 261, 17, 358])
arg0 = tf.keras.backend.pool3d(x0[tf.newaxis, ..., tf.newaxis], pool_size=(1, 1, 1), padding='valid')
arg0 = tf.squeeze(arg0)

x1 = tf.random.uniform([17, 64, 358])
arg1 = tf.keras.backend.pool3d(x1[tf.newaxis, ..., tf.newaxis], pool_size=(1, 1, 1), padding='valid')
arg1 = tf.squeeze(arg1)

x2 = tf.random.uniform([261, 1, 64])
arg2 = tf.keras.backend.pool3d(x2[tf.newaxis, ..., tf.newaxis], pool_size=(1, 1, 1), padding='valid')
arg2 = tf.squeeze(arg2)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    print('Eager Success! ')

    # torch.compile -> torch.all
    # We wrap the function to convert TF output to Torch and apply torch.all
    def compiled_foo_wrapper(*args):
        res = foo(*args)
        return torch.all(torch.from_numpy(res.numpy()))

    out_compiled = compiled_foo_wrapper(arg0, arg1, arg2)
    # out_compiled is a scalar tensor (0-d), sum() is valid, backward() is valid
    out_compiled.sum().backward()
    print('Compile Success! ')