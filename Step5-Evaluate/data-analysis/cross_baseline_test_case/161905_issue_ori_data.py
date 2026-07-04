```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, applications
from typing import Type, Any, Callable, Union, List, Optional

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically

# PyTorch: resnet18(num_classes=NUM_CLASSES)
# TensorFlow: Using ResNet50 as the closest standard equivalent (Keras applications typically start at ResNet50)
base_model = applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES)(x)
model = models.Model(inputs, outputs)

# PyTorch: nn.CrossEntropyLoss()
# TensorFlow: SparseCategoricalCrossentropy with from_logits=True is the equivalent
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model=model.to(device) # Not needed in TF
# model.train() # Not needed explicitly in TF custom loops, handled by 'training=True' argument

# PyTorch: @torch.compile
# TensorFlow: @tf function compiles the function into a graph
@tf.function
def train(images, labels):
    # images=images.to(device) # Not needed
    # labels=labels.to(device) # Not needed
    
    # optimizer.zero_grad() # Not needed in TF

    with tf.GradientTape() as tape:
        # PyTorch: outputs = model(images)
        # TensorFlow: Pass training=True for layers like Dropout/BatchNorm
        outputs = model(images, training=True)
        
        # PyTorch: loss = criterion(outputs, labels)
        # TensorFlow: Loss functions typically take (y_true, y_pred)
        loss = criterion(labels, outputs)
    
    # PyTorch: loss.backward() & optimizer.step()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# PyTorch: torch.randn(BATCH_SIZE, 3, 224, 224)
# TensorFlow: Uses channels_last format (BATCH_SIZE, 224, 224, 3)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))

# PyTorch: torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```