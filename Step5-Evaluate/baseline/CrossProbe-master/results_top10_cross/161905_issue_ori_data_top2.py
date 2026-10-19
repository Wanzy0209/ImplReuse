```python
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.applications import ResNet50
# from typing import Type, Any, Callable, Union, List, Optional # Preserved for structure, though not strictly used in TF snippet

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically (e.g., GPU if available)

# Conversion: PyTorch resnet18 -> TensorFlow ResNet50 (closest standard equivalent)
# Note: ResNet18 is not in standard tf.keras.applications, so ResNet50 is used here.
# Input shape is (224, 224, 3) for TensorFlow (Channel Last) vs (3, 224, 224) for PyTorch.
model = ResNet50(weights=None, classes=NUM_CLASSES, input_shape=(224, 224, 3))

# Conversion: nn.CrossEntropyLoss -> SparseCategoricalCrossentropy
# PyTorch's CrossEntropyLoss expects raw logits, so from_logits=True is set in TF.
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)

# Conversion: optim.SGD -> optimizers.SGD
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model=model.to(device) # Not needed in TensorFlow
# model.train() # Not needed in TensorFlow for custom training loops; handled by 'training=True' argument

# Conversion: @torch.compile -> @tf.function
# tf.function compiles the Python function into a static TensorFlow graph for performance.
@tf.function
def train(images, labels):
    # images=images.to(device) # Not needed
    # labels=labels.to(device) # Not needed

    # Conversion: optimizer.zero_grad() -> Handled implicitly by GradientTape scope
    with tf.GradientTape() as tape:
        # training=True ensures layers like BatchNorm/Dropout behave correctly
        outputs = model(images, training=True)
        
        # Conversion: criterion(outputs, labels) -> criterion(labels, outputs)
        # Note: TF loss functions typically take (y_true, y_pred)
        loss = criterion(labels, outputs)

    # Conversion: loss.backward() -> tape.gradient
    grads = tape.gradient(loss, model.trainable_variables)

    # Conversion: optimizer.step() -> optimizer.apply_gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Conversion: torch.randn -> tf.random.normal
# Note: Shape changed from (BATCH_SIZE, 3, 224, 224) to (BATCH_SIZE, 224, 224, 3) for Channel Last format.
images = tf.random.normal(shape=(BATCH_SIZE, 224, 224, 3))

# Conversion: torch.randint -> tf.random.uniform
labels = tf.random.uniform(shape=(BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```