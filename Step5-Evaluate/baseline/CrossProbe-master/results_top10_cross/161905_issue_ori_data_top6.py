```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.applications import ResNet50
from typing import Type, Any, Callable, Union, List, Optional

# Conversion: PyTorch uses 'mps' for Apple Silicon GPU. 
# TensorFlow automatically handles device placement (GPU/CPU).
BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' 

# Conversion: PyTorch's resnet18. TensorFlow Keras applications typically provide ResNet50.
# We use ResNet50 here as the closest standard equivalent, modifying the top layer for NUM_CLASSES.
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES)(x)
model = models.Model(inputs=base_model.input, outputs=outputs)

# Conversion: nn.CrossEntropyLoss combines LogSoftmax and NLLLoss.
# In TensorFlow, SparseCategoricalCrossentropy with from_logits=True is the equivalent.
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)

# Conversion: optim.SGD maps directly to tf.keras.optimizers.SGD.
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# Conversion: model.to(device) is not needed in TensorFlow.
# model.train() is not needed; training state is handled by the 'training' argument in the call.
# model=model.to(device)
# model.train() 

# Conversion: @torch.compile compiles the Python function into a graph.
# @tf.function does the same for TensorFlow functions.
@tf.function
def train(images, labels):
    # Conversion: Data movement to device is handled automatically by TensorFlow.
    # images=images.to(device)
    # labels=labels.to(device)
    
    # Conversion: optimizer.zero_grad() is implicit in GradientTape.
    with tf.GradientTape() as tape:
        # Conversion: Pass training=True to enable BatchNorm/Dropout behavior.
        outputs = model(images, training=True)
        
        # Conversion: Loss arguments in TF are typically (y_true, y_pred).
        loss = criterion(labels, outputs)

    # Conversion: loss.backward() and optimizer.step() are replaced by gradient calculation and application.
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Conversion: torch.randn generates random normal values.
# Note: TensorFlow uses channels_last format (Batch, Height, Width, Channels) by default.
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))

# Conversion: torch.randint generates random integers.
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```