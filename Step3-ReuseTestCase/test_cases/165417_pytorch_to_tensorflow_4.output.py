import torch
import tensorflow as tf
import numpy as np
import time

# Disable eager execution globally to strictly separate eager/graph modes if desired,
# but in TF 2.x we usually control this via tf.function and tf.config.run_functions_eagerly.
# We will stick to standard TF 2.x behavior where eager is default.

class MyModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.relu = tf.keras.layers.ReLU()

    def __call__(self, x, y):
        # Original API: torch.unique(x, sorted=True, return_inverse=True)
        # Similar API: tf.compat.v1.math.add(x, y)
        # Note: tf.math.add is element-wise and does not produce dynamic shapes based on data values,
        # unlike torch.unique. Therefore, it should not trigger dynamic shape errors in graph mode.
        
        # Perform the operation
        _z = tf.compat.v1.math.add(x, y)
        
        # Mimic the clone/detach and tuple return structure of the original PyTorch model
        _z = tf.identity(_z) 
        return self.relu(_z), _z

def get_input():
    # tf.math.add requires two inputs, unlike torch.unique
    return np.random.randn(8).astype(np.float32), np.random.randn(8).astype(np.float32)

def run_once(model, x, y, label, use_graph=False):
    if use_graph:
        # Compile the call to simulate torch.compile(fullgraph=True)
        compiled_call = tf.function(model.__call__)
        start = time.perf_counter()
        with tf.device('/CPU:0'): # Explicit device to match repro simplicity
            out = compiled_call(x, y)
    else:
        # Run in eager mode
        start = time.perf_counter()
        with tf.device('/CPU:0'):
            out = model(x, y)
            
    elapsed = (time.perf_counter() - start) * 1000
    # Check types of output tensors
    print(f"[{label}] ok, types={[type(t) for t in out]} time={elapsed:.3f}ms")
    return out

def main():
    print("tf.__version__ =", tf.__version__)
    
    model = MyModel()
    x, y = get_input()

    # 1. Run Eager
    print("--- Running Eager ---")
    y_eager = run_once(model, x, y, "Eager")

    # 2. Run Graph (Compiled)
    print("\n--- Running Graph (tf.function) ---")
    y_graph = run_once(model, x, y, "Graph", use_graph=True)

    # 3. Verification
    # Unlike the PyTorch case where torch.compile fails with "Dynamic shape operator",
    # tf.math.add is a standard static-shape operator and should compile successfully.
    # We verify that eager and graph outputs match.
    print("\n--- Verification ---")
    try:
        for i, (eager_tensor, graph_tensor) in enumerate(zip(y_eager, y_graph)):
            np.testing.assert_allclose(eager_tensor.numpy(), graph_tensor.numpy(), rtol=1e-5)
        print("Test Passed: Eager and Graph outputs match.")
    except AssertionError as e:
        print(f"Test Failed: Outputs differ.\n{e}")

if __name__ == "__main__":
    main()