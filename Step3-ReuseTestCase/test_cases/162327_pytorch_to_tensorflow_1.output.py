import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

print("TensorFlow Version:", tf.__version__, flush=True)

# Adapt inputs from the original PyTorch bug report to test tf.experimental.numpy.moveaxis
# Original PyTorch inputs:
# 1. torch.empty((5, 7, 4, 3, 7, 6), dtype=torch.int8)
# 2. torch.empty((4, 9, 2), dtype=torch.int32)
# 3. ()
# 4. False

# Mapping to tf.experimental.numpy.moveaxis(a, source, destination):
# 'a' expects an array. We use the 6D tensor.
input_a = tnp.empty((5, 7, 4, 3, 7, 6), dtype=tnp.int8)

# 'source' expects an int or sequence of ints. We use the 3D tensor (invalid type) to test robustness.
input_source = tnp.empty((4, 9, 2), dtype=tnp.int32)

# 'destination' expects an int or sequence of ints. We use the empty tuple.
input_destination = ()

# Attempt to call the function with potentially mismatched/invalid arguments
# to verify behavior (crash vs. clean exception).
try:
    result = tnp.moveaxis(input_a, input_source, input_destination)
    print("Call succeeded. Result shape:", result.shape)
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")