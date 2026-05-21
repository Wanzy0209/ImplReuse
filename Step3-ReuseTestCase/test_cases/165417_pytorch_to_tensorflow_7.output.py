import torch
import tensorflow as tf
import time

class MyModel(tf.Module):
    def __init__(self):
        super().__init__()
        # Using tf.nn.relu to match the original model's structure
        self.relu = tf.nn.relu

    def __call__(self, x):
        # Adaptation: Using tf.keras.ops.add instead of torch.unique
        # Note: add is a binary operation, so we add x to itself to mimic the transformation step.
        _x = tf.keras.ops.add(x, x)
        _x = tf.identity(_x)  # mimic clone/detach behavior
        
        # Mimic the tuple output structure of the original model (unique_values, inverse_indices)
        # Since add does not produce inverse indices, we return the result and the input.
        return self.relu(_x), x

def GetInput():
    # Generate random input, compatible with CPU or CUDA
    return tf.random.normal((8,))

def run_once(model, x, label):
    t0 = time.perf_counter()
    # TensorFlow does not require torch.no_grad() equivalent for inference unless using GradientTape
    y = model(x)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[{label}] ok, types={[type(t).__name__ for t in (y if isinstance(y, (tuple, list)) else [y])]} "
          f"time={elapsed:.3f}ms")
    return y

def main():
    print("tf.__version__ =", tf.__version__)

    model = MyModel()
    x = GetInput()

    # 1. Eager Mode
    y_eager = run_once(model, x, "Eager")

    # 2. Compiled Mode
    # tf.function is the TensorFlow equivalent of torch.compile.
    # It traces the code to create a graph, similar to fullgraph mode.
    compiled_model = tf.function(model)
    y_compiled = run_once(compiled_model, x, "Compiled (tf.function)")

    # Verify that the compiled output matches the eager output
    # This ensures the API behaves correctly under compilation (unlike the reported bug in PyTorch)
    if isinstance(y_eager, tuple):
        for e, c in zip(y_eager, y_compiled):
            tf.debugging.assert_near(e, c, message="Eager and Compiled outputs differ")
    else:
        tf.debugging.assert_near(y_eager, y_compiled, message="Eager and Compiled outputs differ")

    print("Test passed: tf.keras.ops.add works correctly in both eager and compiled modes.")

if __name__ == "__main__":
    main()