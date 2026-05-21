import torch
import tensorflow as tf
import numpy as np

# Define a simple TensorFlow model mirroring the PyTorch TestModel
class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(20)
        self.fc2 = tf.keras.layers.Dense(1)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_default_model():
    return TestModel()

def get_sample_inputs():
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    with tf.device(device):
        x = tf.random.normal((4, 10))
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    
    # Run in eager mode to get baseline
    original_output = model(*inputs)
    print('Original model output shape:', original_output.shape)

    # In TensorFlow, tf.function is the mechanism to capture/execute graphs.
    # We test the behavior of tf.name_scope (the similar API) inside this graph context.
    @tf.function
    def run_in_graph(model, inputs):
        # The original bug involved torch.compile inside a graph capture.
        # Here we use tf.name_scope inside a tf.function to verify compatibility.
        with tf.name_scope("test_scope"):
            return model(*inputs)

    graph_output = run_in_graph(model, inputs)
    print('Graph output shape:', graph_output.shape)

    # Verify that the outputs are consistent
    assert np.allclose(original_output.numpy(), graph_output.numpy()), "Output mismatch between eager and graph execution"
    print("Test passed: tf.name_scope behaves correctly within graph execution.")

if __name__ == "__main__":
    main()