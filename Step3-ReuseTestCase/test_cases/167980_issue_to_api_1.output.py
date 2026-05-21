import torch
import os
# Reproduce the environment setup from the original issue
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage

import tensorflow as tf
# Import the similar API: tf.experimental.numpy
import tf.experimental.numpy as tnp
from tensorflow.keras import layers, models, datasets

# Setup device context
print("Device: CPU (forced via CUDA_VISIBLE_DEVICES)")

# Load MNIST data
# Note: TensorFlow's MNIST data loading differs slightly from PyTorch's torchvision
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Preprocess data to match PyTorch ToTensor() behavior (0-1 range, channel dim)
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# Create DataLoaders equivalent
batch_size = 80
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Define the Model (PyTorch Net equivalent)
# PyTorch: Flatten -> Linear(784,256) -> ReLU -> Linear(256,256) -> ReLU -> Linear(256,10)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])

# Define Loss and Optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("=== TRAIN ===")
# Training loop (1 batch)
for X, y in train_ds.take(1):
    with tf.GradientTape() as tape:
        # Forward pass
        pred = model(X, training=True)
        loss = loss_fn(y, pred)
    
    # Backward pass
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("=== EVAL ===")
# Evaluation loop (1 batch)
# Leveraging the similar API: tf.experimental.numpy.append
# We use this API to accumulate predictions, mirroring the concatenation logic
# found in the similar API's implementation.
accumulated_preds = None

with tf.device("/CPU:0"):
    for batch, (X, y) in enumerate(test_ds.take(1)):
        print("batch", batch)
        
        # Forward pass (where the crash occurred in the original issue)
        pred = model(X, training=False)
        
        # Use tf.experimental.numpy.append to aggregate results
        # This reuses the logic of the similar API which wraps concatenate.
        if accumulated_preds is None:
            accumulated_preds = pred
        else:
            accumulated_preds = tnp.append(accumulated_preds, pred, axis=0)

# Assertion to verify execution completed (original issue crashed here)
assert accumulated_preds is not None
assert accumulated_preds.shape[0] == batch_size
print("Evaluation completed successfully.")