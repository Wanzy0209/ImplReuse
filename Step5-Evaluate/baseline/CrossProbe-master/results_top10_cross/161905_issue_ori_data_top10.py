```python
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.applications import ResNet50
from typing import Type, Any, Callable, Union, List, Optional

# Conversion: ResNet18 is not in standard tf.keras.applications, using ResNet50 as a substitute
from torchvision.models.resnet import resnet18

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # Conversion: TF handles device placement automatically, explicit device move not required

# Conversion: Using ResNet50 with weights=None and custom classes to match ResNet18 usage
model = ResNet50(weights=None, classes=NUM_CLASSES, input_shape=(224, 224, 3))

# Conversion: nn.CrossEntropyLoss combines LogSoftmax and NLLLoss. 
# In TF, use SparseCategoricalCrossentropy with from_logits=True.
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# Conversion: model.to(device) is implicit in TensorFlow
# Conversion: model.train() is handled by passing training=True to the model call

# Conversion: @torch.compile optimizes the module. @tf_function creates a callable graph for efficiency.
@tf.function
def train(images, labels):
    # Conversion: No explicit .to(device) needed for tensors in TF
    
    # Conversion: optimizer.zero_grad() is not needed in GradientTape workflow
    
    with tf.GradientTape() as tape:
        # Conversion: Pass training=True to ensure layers like BatchNorm/Dropout behave correctly
        outputs = model(images, training=True)
        # Conversion: TF loss functions typically take (y_true, y_pred)
        loss = criterion(labels, outputs)
    
    # Conversion: Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Conversion: Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Conversion: PyTorch uses NCHW format, TensorFlow defaults to NHWC.
# Generating data with shape (Batch, Height, Width, Channels)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))
# Conversion: torch.randint equivalent
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```