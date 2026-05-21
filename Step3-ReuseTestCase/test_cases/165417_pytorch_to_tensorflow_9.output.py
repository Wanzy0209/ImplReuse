import torch
import tensorflow as tf
import numpy as np

class SparseAddModel(tf.Module):
    """
    A simple model that uses tf.sparse.add.
    This API is identified as similar to torch.unique.
    The output shape (number of non-zero elements) depends on the input data values,
    similar to how torch_unique's output depends on input data.
    """
    def __init__(self):
        super().__init__()

    @tf.function
    def __call__(self, a, b):
        # tf.sparse.add performs element-wise addition on sparse tensors.
        # The resulting SparseTensor has a number of non-zero elements that depends
        # on the data (e.g., if values sum to zero, they might be dropped).
        # This mimics the data-dependent shape behavior of torch.unique.
        result = tf.sparse.add(a, b)
        
        # Return components to mimic the tuple output of the original bug report
        # (unique_values, inverse_indices) -> (values, indices)
        return result.values, result.indices

def get_inputs():
    # Create two sparse tensors.
    # We define indices and values such that the addition result is data-dependent.
    # Shape: [2, 2]
    indices_a = [[0, 0], [1, 1]]
    values_a = [1.0, 2.0]
    shape = [2, 2]
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, shape)

    indices_b = [[0, 0], [1, 1]]
    values_b = [3.0, 4.0]
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, shape)
    
    return sp_a, sp_b

def run_once(model, a, b, label):
    print(f"[{label}] Running...")
    try:
        if label == "Eager":
            # Run in eager mode
            vals, inds = model(a, b)
        elif label == "XLA":
            # Run in compiled mode (analogous to torch.compile with fullgraph)
            # We use tf.xla.compile to strictly enforce compilation and check for 
            # dynamic shape issues similar to the PyTorch fullgraph mode.
            compiled_model = tf.xla.compile(model)
            vals, inds = compiled_model(a, b)

        print(f"[{label}] OK. Output values: {vals.numpy()}, Output indices: {inds.numpy()}")
    except Exception as e:
        print(f"[{label}] Failed with error: {type(e).__name__}: {e}")

def main():
    print("TensorFlow version:", tf.__version__)
    
    model = SparseAddModel()
    a, b = get_inputs()

    # 1. Test Eager Mode (Should always work)
    run_once(model, a, b, "Eager")

    # 2. Test Compiled Mode (XLA)
    # This checks if the data-dependent shape of tf.sparse.add is handled
    # correctly under strict compilation, similar to the torch.compile issue.
    run_once(model, a, b, "XLA")

if __name__ == "__main__":
    main()