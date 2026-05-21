import tensorflow as tf
from tensorflow.python.compat.v1 import mixed_precision

# Define a simple model similar to the PyTorch SimpleModel
class SimpleModel(tf.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_size, input_shape=(input_size,))
        self.relu = tf.keras.layers.ReLU()
        self.linear2 = tf.keras.layers.Dense(output_size)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Check for GPU availability (TensorFlow equivalent of backend check)
if tf.config.list_physical_devices('GPU'):
    print(f"Detected {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
    
    model = SimpleModel()
    
    # Use DynamicLossScale as a concrete implementation of the abstract LossScale
    # This adapts the 'DataParallel' wrapper concept to the 'LossScale' wrapper concept
    loss_scale = mixed_precision.DynamicLossScale(
        initial_loss_scale=2**15,
        increment_period=2000,
        multiplier=2.0
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Prepare input data
    batch_size = 20
    input_data = tf.random.normal([batch_size, 10])
    target_data = tf.random.normal([batch_size, 5])

    # Training step to verify LossScale behavior
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            output = model(input_data)
            loss = tf.reduce_mean(tf.square(output - target_data))

        # Apply Loss Scaling
        scaled_loss = loss_scale(loss)
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        
        # Unscale gradients to check for overflow/underflow
        (grads, has_inf_nan) = loss_scale.unscale(grads)
        
        # Update the loss scale based on gradient status
        loss_scale.update(grads)

        # Apply gradients if they are finite
        if not has_inf_nan:
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return loss

    # Execute the step
    output_loss = train_step()
    print("success")
else:
    raise RuntimeError("No GPU detected")