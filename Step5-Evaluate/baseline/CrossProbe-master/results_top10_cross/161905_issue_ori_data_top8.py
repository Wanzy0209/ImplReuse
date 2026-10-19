```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, applications
from typing import Type, Any, Callable, Union, List, Optional

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically based on installation

# PyTorch: resnet18(num_classes=NUM_CLASSES)
# Note: tf.keras.applications does not have ResNet18. Using ResNet50 as the closest standard equivalent.
model = applications.ResNet50(weights=None, classes=NUM_CLASSES)

# PyTorch: nn.CrossEntropyLoss()
# Note: SparseCategoricalCrossentropy with from_logits=True is the TF equivalent
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)

# PyTorch: optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# PyTorch: model=model.to(device); model.train()
# TF: Device placement is automatic. Training mode is handled in the call.
# model.train() is not a method, but we ensure training=True in the forward pass.

# PyTorch: @torch.compile
# TF: @tf.function compiles the function into a graph for performance
@tf.function
def train(images, labels):
    # PyTorch: images=images.to(device); labels=labels.to(device)
    # TF: Device placement is automatic

    # PyTorch: optimizer.zero_grad()
    # TF: Gradients are calculated fresh within GradientTape

    with tf.GradientTape() as tape:
        # PyTorch: outputs = model(images)
        # TF: Pass training=True for BatchNorm/Dropout behavior
        outputs = model(images, training=True)

        # PyTorch: loss = criterion(outputs, labels)
        # TF: Keras losses typically take (y_true, y_pred)
        loss = criterion(labels, outputs)

    # PyTorch: loss.backward()
    # TF: Calculate gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # PyTorch: optimizer.step()
    # TF: Apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# PyTorch: images = torch.randn(BATCH_SIZE, 3, 224, 224)
# Conversion: PyTorch uses (N, C, H, W), TensorFlow uses (N, H, W, C)
images = tf.random.normal(shape=(BATCH_SIZE, 224, 224, 3))

# PyTorch: labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
labels = tf.random.uniform(shape=(BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```