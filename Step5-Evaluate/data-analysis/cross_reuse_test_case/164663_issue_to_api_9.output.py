import tensorflow as tf
import numpy as np

# Define RMSNorm to match the component involved in the original bug
class RMSNorm(tf.Module):
    def __init__(self, d_model, name=None):
        super().__init__(name=name)
        # Fix: use_resource=True ensures ResourceVariable is created, avoiding RefVariable issues with GradientTape
        self.weight = tf.Variable(tf.ones([d_model]), name='weight', use_resource=True)
        self.eps = 1e-5

    def __call__(self, x):
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(variance + self.eps) * self.weight

# Define the model structure similar to the original issue (RMSNorm + Linear)
class TestModel(tf.Module):
    def __init__(self, d_model, name=None):
        super().__init__(name=name)
        self.norm = RMSNorm(d_model, name="norm")
        # Linear layer equivalent
        # Fix: use_resource=True ensures ResourceVariable is created
        self.linear_w = tf.Variable(tf.random.normal([d_model, d_model]), name="linear_w", use_resource=True)
        self.linear_b = tf.Variable(tf.zeros([d_model]), name="linear_b", use_resource=True)

    def __call__(self, x):
        x = self.norm(x)
        x = tf.matmul(x, self.linear_w) + self.linear_b
        return x

def main():
    d_model = 128
    model = TestModel(d_model)

    # Use the Similar API: tf.compat.v1.train.GradientDescentOptimizer
    # This replaces the implicit backward pass in the original PyTorch code
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)

    # Input data
    x = tf.random.normal([16, d_model])

    # Forward pass
    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_sum(y)

    # Compute gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply gradients using the specific API
    # This step corresponds to the backward pass where the original bug occurred
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Assertions to verify the operation completed successfully
    assert loss is not None
    assert all(g is not None for g in grads), "Gradients should not be None"
    print("Test passed: GradientDescentOptimizer successfully handled RMSNorm model.")

if __name__ == "__main__":
    main()