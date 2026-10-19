```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.applications import ResNet50
from typing import Type, Any, Callable, Union, List, Optional

# Note: torchvision.models.resnet18 is not directly available in tf.keras.applications.
# We use ResNet50 as a standard substitute for demonstration.
# from torchvision.models.resnet import resnet18 

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TF handles device placement automatically (e.g. on GPU if available)

# Constructing a ResNet-like model for NUM_CLASSES
# PyTorch: resnet18(num_classes=NUM_CLASSES)
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES)(x)
model = models.Model(inputs, outputs)

# PyTorch: nn.CrossEntropyLoss() combines LogSoftmax and NLLLoss
# TF: SparseCategoricalCrossentropy with from_logits=True is the equivalent
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model=model.to(device) # Not needed in TF
# model.train() # In TF, training mode is handled by passing training=True to the model call

# PyTorch: @torch.compile
# TF: @tf_function compiles the function into a static graph
@tf.function
def train(images, labels):
    # images=images.to(device) # Not needed in TF
    # labels=labels.to(device) # Not needed in TF
    
    # optimizer.zero_grad() # Handled implicitly by GradientTape scope

    with tf.GradientTape() as tape:
        # PyTorch: model(images)
        # TF: model(images, training=True) ensures BatchNorm/Dropout behave correctly
        outputs = model(images, training=True)
        
        # PyTorch: criterion(outputs, labels)
        # TF: criterion(y_true, y_pred) -> Note the argument order difference
        loss = criterion(labels, outputs)

    # loss.backward() # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # optimizer.step() # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# PyTorch: torch.randn(BATCH_SIZE, 3, 224, 224) -> NCHW format
# TF: Uses NHWC format (Batch, Height, Width, Channels)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))

# PyTorch: torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
# TF: tf.random.uniform with dtype int32
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```