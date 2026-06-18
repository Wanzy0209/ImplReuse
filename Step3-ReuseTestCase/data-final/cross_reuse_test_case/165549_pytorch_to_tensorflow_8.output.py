import torch
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

def test_sparse_conditional_accumulator_shape():
    """
    Adapted test case for tf.raw_ops.SparseConditionalAccumulator based on 
    PyTorch issue #165549 (abs returning empty tensors).
    
    The core bug reproduction logic involves verifying that an operation
    returns an object with the expected shape configuration, rather than
    defaulting to an empty or zero-sized state.
    """
    # Define expected shape and type
    expected_shape = [4, 4]
    dtype = tf.float32

    # Create the accumulator
    # In PyTorch: result = torch.abs(t)
    # In TF: We instantiate the accumulator which should hold the shape configuration.
    # Note: We use the class wrapper to access the .shape property, 
    # mirroring the check on the result object in PyTorch.
    accumulator = data_flow_ops.SparseConditionalAccumulator(
        dtype=dtype,
        shape=expected_shape,
        name="test_accumulator"
    )

    # Verify the shape
    # In PyTorch: assert result.shape == torch.Size([4, 4])
    # Here we check if the accumulator object correctly reports the shape 
    # passed during construction.
    assert accumulator.shape == tf.TensorShape(expected_shape), \
        f"Expected accumulator shape {expected_shape}, but got {accumulator.shape}"

if __name__ == "__main__":
    test_sparse_conditional_accumulator_shape()
    print("Test passed.")