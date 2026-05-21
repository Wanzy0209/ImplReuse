import torch
import tensorflow as tf

# Constants
BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01

# Setup Device
# Note: On macOS with MPS support, TensorFlow typically exposes this as a GPU.
# We attempt to use the GPU if available, otherwise fallback to CPU.
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Model Definition
# Using ResNet50 as the standard equivalent to ResNet18 in Keras applications
model = tf.keras.applications.ResNet50(weights=None, classes=NUM_CLASSES)

# Optimizer and Loss
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
criterion = tf.keras.losses.SparseCategoricalCrossentropy()

# Adaptation: Using tf.keras.name_scope to wrap the training logic
# This replaces the @torch.compile decorator in the original PyTorch code.
# While name_scope is for graph organization rather than compilation,
# we use it here to group the training step operations as requested.
def train_step(images, labels):
    with tf.device(device):
        with tf.keras.name_scope("training_step"):
            with tf.GradientTape() as tape:
                # Forward pass
                outputs = model(images, training=True)
                # Loss calculation
                loss = criterion(labels, outputs)
            
            # Backward pass (equivalent to loss.backward())
            gradients = tape.gradient(loss, model.trainable_variables)
            # Optimizer step (equivalent to optimizer.step())
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return loss

# Generate dummy data
# Note: TensorFlow uses channels_last (NHWC) format by default
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

# Execute the test case
print(f"Running test on device: {device}")
try:
    loss_value = train_step(images, labels)
    
    # Assertions to verify the behavior
    assert loss_value is not None, "Loss calculation failed"
    assert not tf.math.is_nan(loss_value), "Loss resulted in NaN"
    assert not tf.math.is_inf(loss_value), "Loss resulted in Inf"
    
    print(f"Test passed successfully. Loss: {loss_value.numpy()}")
except Exception as e:
    print(f"Test failed with error: {e}")
    raise