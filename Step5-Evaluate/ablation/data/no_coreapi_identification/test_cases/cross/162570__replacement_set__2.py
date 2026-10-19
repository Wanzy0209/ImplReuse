import torch
import tensorflow as tf
from tensorflow.keras import constraints

# nn.Module -> tf.keras.constraints.serialize
# We define a constraint to act as the model component
net = constraints.MaxNorm()

# torch.tensor -> torch.isinf
x = torch.isinf([0.0])
y = torch.isinf([0.0])

# torch.onnx.export -> tf.keras.constraints.serialize
# We serialize the constraint instead of exporting to ONNX
tf.keras.constraints.serialize(net)

# ort.InferenceSession -> tf.keras.constraints.serialize
# We assign the serialized config to sess
sess = tf.keras.constraints.serialize(net)

# torch.atan2 -> torch.ne
# We perform the not-equal operation on the inputs
torch_result = torch.ne(x, y)

# sess.run -> tf.keras.constraints.serialize
# x.numpy -> tf.keras.constraints.serialize
# y.numpy -> tf.keras.constraints.serialize
# We call serialize again. Note: The inputs x and y are not passed to the function
# as the API signature does not support them, preserving runnability.
ort_result = tf.keras.constraints.serialize(net)

print("torch_result", torch_result)
print("ort_result", ort_result)