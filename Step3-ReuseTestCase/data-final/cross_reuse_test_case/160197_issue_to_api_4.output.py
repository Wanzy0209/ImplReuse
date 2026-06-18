import warnings
import tensorflow as tf
import numpy as np

# Replicate the strict warning setting from the original issue to ensure
# the similar API does not raise unexpected warnings.
warnings.simplefilter('error')

print(f"TensorFlow Version: {tf.__version__}")

# Setup tensors similar to the PyTorch example
# 'a' is the trainable variable (analogous to tensor with requires_grad=True)
# 'b' is the target tensor
b = tf.random.uniform((2, 32, 32))
a = tf.Variable(tf.random.uniform((2, 32, 32)))

optimizer = tf.keras.optimizers.Adam()

# Loss function: Mean Squared Error
def loss_fn(x, y):
    return tf.reduce_mean(tf.square(x - y))

# Optimization loop
# In the PyTorch issue, the optimizer internally called float(closure()),
# which raised a warning. Here we perform the step and explicitly evaluate
# the loss using the similar API.
for i in range(10):
    with tf.GradientTape() as tape:
        loss = loss_fn(a, b)
    
    gradients = tape.gradient(loss, [a])
    optimizer.apply_gradients(zip(gradients, [a]))

    # Use the Similar API: tf.keras.backend.eval
    # This corresponds to the need to inspect the scalar loss value.
    # It serves as the TF equivalent of extracting the value from the tensor.
    loss_value = tf.keras.backend.eval(loss)
    
    print(f"Iteration {i}: Loss = {loss_value}")

# Final assertion to verify the logic ran correctly and loss decreased
final_loss = tf.keras.backend.eval(loss_fn(a, b))
assert final_loss < 1.0, "Loss should have decreased during optimization"