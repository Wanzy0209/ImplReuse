import torch
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras

# Replacing nn.Module with tf.keras.quantizers.serialize
# Since serialize is a function, we define a function to represent the 'Net'
def net(x, y):
    # Replacing torch.atan2 with torch.igamma
    return torch.igamma(x, y)

# Replacing torch.tensor with torch.nn.functional.pdist
# We need a 2D tensor for pdist. Creating a tensor of zeros to get zero output.
base_tensor = torch.tensor([[0.0], [0.0]])
x = F.pdist(base_tensor)
y = F.pdist(base_tensor)

# Replacing torch.onnx.export with tf.keras.quantizers.serialize
# We need a quantizer object to serialize.
class DummyQuantizer(keras.quantizers.Quantizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return inputs
    def get_config(self):
        return super().get_config()

# "Export" step
export_result = tf.keras.quantizers.serialize(DummyQuantizer())

# Replacing ort.InferenceSession with tf.keras.quantizers.serialize
# Replacing sess.run with tf.keras.quantizers.serialize
# Replacing x.numpy with tf.keras.quantizers.serialize
# Replacing y.numpy with tf.keras.quantizers.serialize

# We perform the "run" by calling serialize again.
# We also mimic the input dictionary construction by calling serialize on the inputs (conceptually, though serialize expects a quantizer).
# To keep it runnable, we will just call serialize on the dummy quantizer.
ort_result = tf.keras.quantizers.serialize(DummyQuantizer())

# Run the PyTorch model
torch_result = net(x, y)

# Also test torch.xlogy (replacement for torch.angle mentioned in bug description)
# xlogy(0, 0) should be 0
xlogy_result = torch.xlogy(x, y)

print("torch_result (igamma)", torch_result)
print("ort_result (serialize)", ort_result)
print("xlogy_result", xlogy_result)

# Assertions based on expected behavior
# torch.igamma(0, 0) is NaN
assert torch.isnan(torch_result).all(), "igamma(0, 0) should be NaN"
# torch.xlogy(0, 0) is 0
assert torch.eq(xlogy_result, 0.0).all(), "xlogy(0, 0) should be 0"