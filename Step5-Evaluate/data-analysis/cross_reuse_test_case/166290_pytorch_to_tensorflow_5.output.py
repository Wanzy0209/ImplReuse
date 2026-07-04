import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(974450504)
np.random.seed(974450504)

def fuzzed_program(arg_0, arg_1):
    """
    Adapted from PyTorch test case to TensorFlow.
    Preserves the tensor manipulation logic (chunk, gather, squeeze)
    and integrates the similar API: tf.keras.optimizers.SGD.
    """
    # var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # TensorFlow equivalent: tf.split
    var_node_2 = tf.split(arg_0, 3, axis=3)[0] # size=(17, 30, 17, 1)

    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    var_node_5 = tf.fill((17,), tf.cast(3, tf.int64)) # size=(17,), dtype=int64

    # _input_size_var_node_4 = var_node_5.size(0)
    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), ...)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)

    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0) # size=(15,)

    # _input_size_var_node_1 = var_node_2.size(0)
    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), ...)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)

    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # TensorFlow equivalent: tf.gather
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1)

    # var_node_0 = torch.squeeze(var_node_1)
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17)

    # --- Testing the Similar API: tf.keras.optimizers.SGD ---
    # We use the generated tensor var_node_0 as a target to perform an SGD step.
    # This verifies that the optimizer handles the data flow resulting from the squeeze operation.

    # Create a trainable variable with the same shape as the squeezed tensor
    var_to_train = tf.Variable(tf.random.normal_initializer()(shape=tf.shape(var_node_0), dtype=tf.float32), name='weights')
    
    # Instantiate the SGD optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Cast boolean tensor to float for loss calculation
    target_tensor = tf.cast(var_node_0, tf.float32)

    # Calculate gradients and apply updates
    with tf.GradientTape() as tape:
        # Dummy loss: Mean Squared Error
        loss = tf.reduce_mean(tf.square(var_to_train - target_tensor))
    
    grads = tape.gradient(loss, [var_to_train])
    optimizer.apply_gradients(zip(grads, [var_to_train]))

    return loss, var_to_train

# --- Input Setup ---
# arg_0: size=(17, 30, 17, 3), dtype=bool
# Replacing torch.as_strided with direct generation for TensorFlow compatibility
arg_0 = tf.cast(tf.random.uniform((17, 30, 17, 3)) > 0.5, tf.bool)

# arg_1: size=(15,), dtype=int64
arg_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)

# --- Execution ---

# 1. Eager Execution
print("Running Eager Execution...")
loss_eager, weights_eager = fuzzed_program(arg_0, arg_1)
print(f" Eager success. Loss: {loss_eager.numpy()}")

# 2. Compiled Execution (tf.function)
# This mimics the torch.compile behavior to check for graph compilation issues
print("Running Compiled Execution...")
compiled_program = tf.function(fuzzed_program)
loss_compiled, weights_compiled = compiled_program(arg_0, arg_1)
print(f" Compile success. Loss: {loss_compiled.numpy()}")

# --- Verification ---
# Check if results match between eager and compiled modes
assert np.allclose(loss_eager.numpy(), loss_compiled.numpy()), "Loss mismatch!"
assert np.allclose(weights_eager.numpy(), weights_compiled.numpy()), "Weights mismatch!"
print(" Verification passed: Eager and Compiled results are consistent.")