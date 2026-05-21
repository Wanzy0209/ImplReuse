import tensorflow as tf
import numpy as np

# Constants adapted from the original PyTorch test case
BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01

# 1. Model Setup
# Using ResNet50 as the TensorFlow equivalent to ResNet18 for this test case
# weights=None ensures it runs without downloading pre-trained weights
inputs = tf.keras.Input(shape=(224, 224, 3))
base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model(inputs, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)
model = tf.keras.Model(inputs, outputs)

# 2. Optimizer and Loss setup
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 3. Data generation
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
labels = tf.random.uniform((BATCH_SIZE,), maxval=NUM_CLASSES, dtype=tf.int32)

# 4. Training Step using the Similar API: tf.name_scope
# The original bug occurred during the backward pass inside a compiled function.
# Here we wrap the training logic (forward and backward) within tf.name_scope
# to verify that the operations execute correctly within the named context.
@tf.function # Acts as the compilation mechanism in TF
def train_step(images, labels):
    # Using tf.name_scope to group operations, analogous to the context in the original bug
    with tf.name_scope("training_step"):
        with tf.GradientTape() as tape:
            # Forward pass
            with tf.name_scope("forward"):
                outputs = model(images, training=True)
                loss = loss_fn(labels, outputs)
        
        # Backward pass (Gradient Computation)
        # This corresponds to the loss.backward() call in the original bug report
        with tf.name_scope("backward"):
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
    return loss

# 5. Execution and Verification
print("Running training step with tf.name_scope...")
try:
    loss_value = train_step(images, labels)
    
    # Assertions to verify successful execution
    assert loss_value is not None, "Loss value is None"
    assert not tf.math.is_nan(loss_value), "Loss is NaN"
    assert not tf.math.is_inf(loss_value), "Loss is Inf"
    
    print(f"Test passed successfully. Loss: {loss_value.numpy()}")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise