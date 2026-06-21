import torch
import tensorflow as tf
import numpy as np

# Setup: Ensure we are in Graph Mode (TF 1.x default)
# This creates the "Context" that is incompatible with the API call.
tf.compat.v1.disable_v2_behavior()

class TestModel:
    def __init__(self):
        self.fc1 = tf.compat.v1.layers.Dense(20)
        self.fc2 = tf.compat.v1.layers.Dense(1)

    def __call__(self, x):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

def get_default_model():
    return TestModel()

def get_sample_inputs():
    # Use numpy to avoid triggering TF graph construction prematurely
    x = np.random.randn(4, 10).astype(np.float32)
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()

    # In TF 1.x, we are in a graph context by default.
    # The original bug involves calling torch.compile inside a CUDA Graph capture.
    # Here, we test calling enable_eager_execution inside a Graph context.
    
    try:
        # This mimics the "torch.compile" call inside the graph capture context.
        # It attempts to switch execution modes while the graph is active.
        tf.compat.v1.enable_eager_execution()
        
        # If we reach here, the behavior is unexpected
        print("enable_eager_execution succeeded unexpectedly.")
        
    except RuntimeError as e:
        # This is the expected outcome, similar to the PyTorch bug
        print(f"Caught expected error: {e}")
        assert "cannot be enabled" in str(e).lower()

if __name__ == "__main__":
    main()