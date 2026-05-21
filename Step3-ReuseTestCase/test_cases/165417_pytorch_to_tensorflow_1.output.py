import torch
import tensorflow as tf
import time

class MyModel(tf.Module):
    def __init__(self):
        super().__init__()
        # No specific layers needed for concat, just the op

    def forward(self, x, y):
        # Using tf.concat as the similar API to torch.unique
        # Note: tf.concat takes a list of tensors and an axis
        result = tf.concat([x, y], axis=0)
        return result

def GetInput():
    # Generate random tensors similar to torch.randn(8)
    # Splitting into two parts to concatenate
    return tf.random.normal((4,)), tf.random.normal((4,))

def run_once(model, x, y, label):
    t0 = time.perf_counter()
    # Execute the model
    result = model(x, y)
    print(f"[{label}] ok, type={type(result)} time={(time.perf_counter()-t0)*1000:.3f}ms")
    return result

def main():
    print("tf.__version__ =", tf.__version__)

    model = MyModel()
    x, y = GetInput()

    # 1. Eager Mode
    # In TensorFlow, calling a tf.Module method directly executes in eager mode by default
    print("--- Eager Mode ---")
    run_once(model.forward, x, y, "Eager")

    # 2. Compiled Mode (Graph Mode)
    # tf.function is the TensorFlow equivalent of torch.compile
    # It traces the operations to create a static graph
    print("--- Compiled Mode (tf.function) ---")
    compiled_model = tf.function(model.forward)
    run_once(compiled_model, x, y, "Compiled")

if __name__ == "__main__":
    main()