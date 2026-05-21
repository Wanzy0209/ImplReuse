import tensorflow as tf
import numpy as np

# Enable eager execution (The API under test)
# This must be called at the very beginning of the program.
tf.compat.v1.enable_eager_execution()

# Constants
BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01

# Define Model (ResNet equivalent)
# We use ResNet50 from Keras as a proxy for ResNet18 to maintain similar complexity.
# weights=None ensures random initialization.
model = tf.keras.applications.ResNet50(weights=None, classes=NUM_CLASSES, include_top=True)

# Loss and Optimizer
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# Training function
# In eager execution, we use tf.GradientTape to track operations for automatic differentiation.
def train(images, labels):
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(images, training=True)
        # Loss calculation
        loss = criterion(labels, outputs)
    
    # Backward pass (calculate gradients)
    gradients = tape.gradient(loss, model.trainable_variables)
    # Optimizer step (apply gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Create dummy data
# ResNet50 expects input shape (Batch_Size, Height, Width, Channels)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
labels = tf.random.uniform((BATCH_SIZE,), maxval=NUM_CLASSES, dtype=tf.int32)

# Run the training step
# This verifies that eager execution allows for the forward and backward passes
# to run imperatively, similar to the expected behavior of the PyTorch code.
loss_value = train(images, labels)

# Assertion to verify the test case ran successfully
assert loss_value is not None
print(f"Test passed. Loss: {loss_value.numpy()}")