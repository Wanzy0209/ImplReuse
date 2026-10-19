```python
import tensorflow as tf
from tensorflow.keras import losses, optimizers
# Note: TensorFlow Keras Applications does not have ResNet18, using ResNet50 as the closest equivalent.
from tensorflow.keras.applications import ResNet50

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically (GPU/CPU)

# Initialize model. weights=None for random initialization.
# input_shape=(224, 224, 3) matches the data generation below (NHWC format).
model = ResNet50(weights=None, classes=NUM_CLASSES, input_shape=(224, 224, 3))

criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model.train() is handled by passing training=True to model() in TF
# model.to(device) is handled automatically by TF

# @torch.compile equivalent in TensorFlow is @tf.function
@tf.function
def train(images, labels):
    # No explicit .to(device) needed in TF, data is usually on the same device as the model or handled automatically

    with tf.GradientTape() as tape:
        # training=True is crucial for layers like BatchNorm, Dropout
        outputs = model(images, training=True)
        # TF loss functions typically take (y_true, y_pred)
        loss = criterion(labels, outputs)

    # Calculate gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Apply gradients (equivalent to optimizer.step())
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Data generation
# PyTorch uses NCHW (Batch, Channels, Height, Width)
# TensorFlow uses NHWC (Batch, Height, Width, Channels)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```