```python
import tensorflow as tf
from tensorflow.keras import losses, optimizers
# Using ResNet50 as the closest standard equivalent to torchvision's resnet18
from tensorflow.keras.applications import ResNet50

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically based on available hardware

# PyTorch: resnet18(num_classes=NUM_CLASSES)
# Note: tf.keras.applications does not have ResNet18, using ResNet50 as substitute
model = ResNet50(weights=None, classes=NUM_CLASSES)

# PyTorch: nn.CrossEntropyLoss()
# Note: from_logits=True is required because ResNet outputs raw logits
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# PyTorch: model.train()
# In Keras, training mode is enabled via the 'training' argument in the call
# model.train() is not a standalone method

# PyTorch: @torch.compile
# TensorFlow: @tf.function for graph optimization
@tf.function
def train(images, labels):
    # PyTorch: images=images.to(device); labels=labels.to(device)
    # TF handles device placement automatically

    # PyTorch: optimizer.zero_grad()
    # TF optimizers handle gradient reset internally

    with tf.GradientTape() as tape:
        # PyTorch: outputs = model(images)
        # training=True is crucial for layers like BatchNorm, Dropout
        outputs = model(images, training=True)

        # PyTorch: loss = criterion(outputs, labels)
        # TF loss signature is (y_true, y_pred)
        loss = criterion(labels, outputs)

    # PyTorch: loss.backward()
    # PyTorch: optimizer.step()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# PyTorch: torch.randn(BATCH_SIZE, 3, 224, 224)
# Note: TensorFlow uses channels-last format (NHWC) by default
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))

# PyTorch: torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```