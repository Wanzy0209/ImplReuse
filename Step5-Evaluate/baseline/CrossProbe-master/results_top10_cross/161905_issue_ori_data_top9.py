```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from typing import Type, Any, Callable, Union, List, Optional

# Note: TensorFlow does not have a built-in ResNet18 in keras.applications.
# We use ResNet50 as the closest standard equivalent for this translation.
from tensorflow.keras.applications import ResNet50

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement (GPU/CPU) automatically

# Constructing the model equivalent to resnet18
# We use ResNet50 here as a standard substitute, with random weights (weights=None)
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES)(x) # No activation, logits for CrossEntropy
model = models.Model(inputs=base_model.input, outputs=outputs)

# nn.CrossEntropyLoss in PyTorch combines LogSoftmax and NLLLoss.
# In TensorFlow, SparseCategoricalCrossentropy with from_logits=True is the equivalent.
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model.to(device) is not required in TF; it handles device placement automatically.
# model.train() is not a persistent state in TF; it is handled via the 'training' argument in the call.

# @torch.compile optimizes the graph. In TensorFlow, @tf.function serves a similar purpose.
@tf.function
def train(images, labels):
    # images=images.to(device) # Not needed, TF handles data placement
    # labels=labels.to(device) # Not needed

    # optimizer.zero_grad() is handled implicitly by the GradientTape context

    with tf.GradientTape() as tape:
        # training=True is required for layers like BatchNorm or Dropout
        outputs = model(images, training=True)
        # Note: TF loss functions typically take (y_true, y_pred)
        loss = criterion(labels, outputs)

    # loss.backward() and optimizer.step()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# PyTorch uses Channels First (N, C, H, W) by default.
# TensorFlow uses Channels Last (N, H, W, C) by default.
# Adjusting shape for tf.random.normal: (BATCH_SIZE, 224, 224, 3)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```