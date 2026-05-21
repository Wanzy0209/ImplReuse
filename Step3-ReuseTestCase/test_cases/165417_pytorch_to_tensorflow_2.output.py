import torch
import tensorflow as tf
import numpy as np

class MyModel(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function # Equivalent to torch.compile
    def __call__(self, sp_a, sp_b):
        # Using tf.sparse.concat as the target API
        # This operation's output shape depends on the input shapes.
        # We are testing the compilation behavior here.
        result = tf.sparse.concat(axis=0, values=[sp_a, sp_b])
        return result

def get_input():
    # Create sparse tensors
    indices_a = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values_a = np.array([1, 2], dtype=np.float32)
    dense_shape_a = np.array([3, 4], dtype=np.int64)
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values_b = np.array([3, 4], dtype=np.float32)
    dense_shape_b = np.array([3, 4], dtype=np.int64)
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)
    return sp_a, sp_b

def run_once(model, sp_a, sp_b, label):
    try:
        if label == "Eager":
            # Direct call for eager execution
            y = tf.sparse.concat(axis=0, values=[sp_a, sp_b])
        else:
            # Call the compiled model
            y = model(sp_a, sp_b)

        print(f"[{label}] ok, type={type(y)}, shape={y.shape}")
        return y
    except Exception as e:
        print(f"[{label}] Error: {e}")
        return None

def main():
    print("tf.__version__ =", tf.__version__)

    model = MyModel()
    sp_a, sp_b = get_input()

    # Run Eager
    y_eager = run_once(model, sp_a, sp_b, "Eager")

    # Run Graph (Compiled)
    y_graph = run_once(model, sp_a, sp_b, "Graph")

    # Verify
    if y_eager is not None and y_graph is not None:
        # Check if shapes match
        assert y_eager.shape == y_graph.shape
        # Check values
        assert tf.reduce_all(y_eager.values == y_graph.values).numpy()
        print("Test Passed: Eager and Graph outputs match.")

if __name__ == "__main__":
    main()