import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta

# Basic container for variables to mimic TestDataset structure
class TestModel:
    def __init__(self, w: tf.Tensor):
        self.w = tf.Variable(w)

    def __call__(self):
        return self.w

def test_bug(device: str = '/GPU:0'):
    # TensorFlow uses device contexts instead of a global set_default_device
    # We mimic the behavior by wrapping the logic in the device context.
    with tf.device(device):
        # Create data (mimicking x and y tensors)
        w = tf.random.normal((100, 3))
        
        model = TestModel(w)

        # Instantiate Optimizer (mimicking random_split)
        # Adadelta is an optimizer, so we test its update step
        optimizer = Adadelta(learning_rate=1.0)

        # Define a simple loss to generate gradients
        def loss():
            return tf.reduce_sum(model.w ** 2)

        # Perform optimization step (The operation being tested)
        # In the original bug, this step failed.
        with tf.GradientTape() as tape:
            loss_value = loss()
        
        grads = tape.gradient(loss_value, [model.w])
        optimizer.apply_gradients(zip(grads, [model.w]))

        print(f"Device {device} worked.")

# Run tests
test_bug(device='/CPU:0') # works

# Check for GPU availability before testing
if tf.config.list_physical_devices('GPU'):
    test_bug(device='/GPU:0') # Check if this throws an error
else:
    print("GPU not available, skipping '/GPU:0' test.")