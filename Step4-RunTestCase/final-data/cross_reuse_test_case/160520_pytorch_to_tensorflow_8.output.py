import sys
import tempfile

# Attempt to import dependencies
try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle the specific GLIBCXX environment error or general missing dependencies
    error_msg = str(e)
    if "GLIBCXX" in error_msg or "libstdc++" in error_msg:
        print(f"Test skipped: Incompatible system library (GLIBCXX) detected. Error: {e}")
    else:
        print(f"Test skipped: Required library not found. Error: {e}")
    sys.exit(0)

# Define a simple MLP model similar to the PyTorch example
class SimpleMLP(tf.keras.Model):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = tf.reshape(x, [tf.shape(x)[0], 28 * 28])
        x = tf.nn.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model
mlp = SimpleMLP()

# Configure GPU if available to match the .cuda() behavior in the original test
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Setup Profiler to monitor execution overhead
log_dir = tempfile.mkdtemp()
tf.profiler.experimental.start(log_dir)

# The core logic: Run inside the specific context manager (tf.keras.name_scope)
# This replaces the torch.device context from the original bug report to test
# the behavior of the similar API.
with tf.keras.name_scope("inference_scope"):
    for i in range(10):
        x = tf.random.normal((i, 28, 28))
        y = mlp(x)
        loss = tf.reduce_sum(y)

tf.profiler.experimental.stop()
print(f"Profile trace saved to: {log_dir}")