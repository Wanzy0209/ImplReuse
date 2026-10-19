```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
# Note: tf.keras.applications does not strictly have ResNet18. 
# Using ResNet50 as the closest standard equivalent for demonstration.
from tensorflow.keras.applications import ResNet50

BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 0.01
# device='mps' # TensorFlow handles device placement automatically (e.g., GPU if available)

# PyTorch: resnet18(num_classes=NUM_CLASSES)
# Constructing a ResNet-like model with 10 output classes
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES)(x) # Logits
model = models.Model(inputs=base_model.input, outputs=outputs)

# PyTorch: nn.CrossEntropyLoss() combines LogSoftmax and NLLLoss
# TF: SparseCategoricalCrossentropy with from_logits=True
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)

# PyTorch: optim.SGD
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

# model=model.to(device) # Not needed in TF
# model.train() # In TF, training mode is handled by passing training=True to the call

# PyTorch: @torch.compile
# TF: @tf.function (Graph compilation)
@tf.function
def train(images, labels):
    # images=images.to(device) # Not needed in TF
    # labels=labels.to(device) # Not needed in TF
    
    # optimizer.zero_grad() # Not needed in TF GradientTape

    with tf.GradientTape() as tape:
        # PyTorch: model(images)
        # TF: model(images, training=True) to enable dropout/batchnorm updates
        outputs = model(images, training=True)
        
        # PyTorch: criterion(outputs, labels)
        # TF: criterion(y_true, y_pred)
        loss = criterion(labels, outputs)
    
    # PyTorch: loss.backward() + optimizer.step()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# PyTorch: torch.randn(BATCH_SIZE, 3, 224, 224) -> NCHW format
# TF: Uses NHWC format (Batch, Height, Width, Channels)
images = tf.random.normal((BATCH_SIZE, 224, 224, 3))

# PyTorch: torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
labels = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

train(images, labels)
```