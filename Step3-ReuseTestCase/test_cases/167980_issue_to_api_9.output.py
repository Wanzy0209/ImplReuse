import os
# Mimic the environment setup from the original bug report
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

# Check device availability
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print("Device:", device)

# Load MNIST data
print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension to match image tensor expectations (Batch, Height, Width, Channel)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

batch_size = 80

# Create TensorFlow Datasets to mimic PyTorch DataLoader
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Define the Model
class Net(models.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dense3 = layers.Dense(10)

    def call(self, x, training=None):
        # LEVERAGE SIMILAR API: tf.pad
        # The original bug involved a crash during eval. We use tf.pad here to test
        # tensor manipulation stability during the train-to-eval transition.
        # Padding the 28x28 input to 30x30.
        x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

model = Net()
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=0.01)

print("=== TRAIN ===")
# Training loop (one batch)
for X, y in train_ds.take(1):
    with tf.GradientTape() as tape:
        pred = model(X, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Train Loss: {loss.numpy()}")
    break

print("=== EVAL (CRASH CHECK) ===")
# Evaluation loop (one batch)
# The original PyTorch issue crashed here. This test verifies that the 
# similar API (tf.pad) and the model structure handle this transition correctly.
for batch, (X, y) in enumerate(test_ds.take(1)):
    print("batch", batch)
    pred = model(X, training=False)
    
    # Assertions to verify execution completed successfully
    assert pred is not None, "Model output is None"
    assert pred.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {pred.shape}"
    print("Eval Output Shape:", pred.shape)
    break

print("Test Passed: No crash during evaluation.")