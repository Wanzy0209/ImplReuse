```python
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, models
from tensorflow.keras.applications import ResNet18
from typing import Type, Any, Callable, Union, List, Optional

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically or via tf.device context

# Initialize ResNet18. 
# Note: weights=None initializes random weights (matching PyTorch default).
# classes=NUM_CLASSES adjusts the final dense layer.
model = ResNet18(weights=None, classes=NUM_CLASSES, input_shape=(224, 224, 3))

# PyTorch CrossEntropyLoss combines LogSoftmax and NLLLoss.
# In TF, SparseCategoricalCrossentropy with from_logits=True does the same.
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model.to(device) is not required in TensorFlow
# model.train() is implicit when passing training=True to the model call

@tf.function # Equivalent to torch.compile for graph optimization
def train(images, labels):
    # Data is already on the correct device in TF (or handled implicitly)
    
    # optimizer.zero_grad() is handled by the GradientTape context
    
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(images, training=True)
        # Loss calculation
        loss = criterion(labels, outputs)

    # Backward pass (calculate gradients)
    gradients = tape.gradient(loss, model.trainable_variables)
    # Optimizer step (apply gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Data generation
# PyTorch uses NCHW format (Batch, Channel, Height, Width)
# TensorFlow uses NHWC format (Batch, Height, Width, Channel)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```