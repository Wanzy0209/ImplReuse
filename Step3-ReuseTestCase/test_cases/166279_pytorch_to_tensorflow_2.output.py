import torch
import tensorflow as tf
from tensorflow.keras import backend as K

# Set seed for reproducibility
tf.random.set_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Replicate tensor manipulations from the original PyTorch code
    # var_node_3 = torch.full((12,), False, dtype=torch.bool)
    var_node_3 = tf.fill((12,), False)

    # var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    # tf.split returns a list, we take the first chunk
    var_node_2 = tf.split(var_node_3, 4, axis=0)[0]

    var_node_6 = arg_0
    var_node_7 = arg_1

    # _input_size_var_node_5 = var_node_6.size(0)
    _input_size_var_node_5 = tf.shape(var_node_6)[0]

    # _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    _index_var_node_5 = tf.random.uniform((10,), minval=0, maxval=_input_size_var_node_5, dtype=tf.int32)

    # var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    var_node_5 = tf.gather(var_node_6, _index_var_node_5, axis=0)

    # var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    var_node_4 = tf.split(var_node_5, 2, axis=0)[0]

    var_node_10 = arg_2
    # var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    var_node_9 = tf.split(var_node_10, 4, axis=1)[0]

    # var_node_8 = torch.squeeze(var_node_9)
    var_node_8 = tf.squeeze(var_node_9)

    # var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    var_node_1 = tf.concat([var_node_2, var_node_4, var_node_8], axis=0)

    var_node_11 = arg_3
    # var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    var_node_0 = tf.concat([var_node_1, var_node_11], axis=0)

    # --- Adaptation for tf.keras.backend.resize_volumes ---
    # The API expects a 5D tensor. var_node_0 is 1D (size 16).
    # We reshape it to (Batch, Depth, Height, Width, Channels) = (1, 1, 1, 16, 1)
    # to fit the API requirements while preserving the data flow.
    input_5d = tf.reshape(var_node_0, (1, 1, 1, 16, 1))

    # Call the similar API: resize_volumes
    # resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
    # We use factors > 1 to ensure resizing happens.
    resized_volume = K.resize_volumes(input_5d, depth_factor=2, height_factor=2, width_factor=2, data_format='channels_last')

    # Ensure gradient computation by multiplying with sentinel
    result = resized_volume * sentinel
    return result

# Sentinel variable to ensure gradient computation
sentinel = tf.Variable(1.0)

# Create arguments mimicking the original shapes and types
# arg_0: size=(12,), dtype=bool
arg_0 = tf.cast(tf.random.uniform((12,), 0, 2, dtype=tf.int32), tf.bool)
# arg_1: size=(10,), dtype=int64
arg_1 = tf.cast(tf.random.uniform((10,), 5, 30, dtype=tf.int32), tf.int64)
# arg_2: size=(6, 4), dtype=bool
arg_2 = tf.cast(tf.random.uniform((6, 4), 0, 2, dtype=tf.int32), tf.bool)
# arg_3: size=(2,), dtype=bool
arg_3 = tf.cast(tf.random.uniform((2,), 0, 2, dtype=tf.int32), tf.bool)

args = (arg_0, arg_1, arg_2, arg_3, sentinel)

# 1. Eager Execution
print("Running Eager Execution...")
result_eager = fuzzed_program(*args)
print(f" Eager success. Shape: {result_eager.shape}")

# 2. Graph/Compiled Execution (tf.function)
print("Running Compiled Execution (tf.function)...")
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print(f" Compile success. Shape: {result_compiled.shape}")

# 3. Verification
# Check if results are close
if tf.reduce_all(tf.abs(result_eager - result_compiled) < 1e-5).numpy():
    print(" Eager and Compiled results match.")
else:
    print(" Divergence detected between Eager and Compiled modes.")