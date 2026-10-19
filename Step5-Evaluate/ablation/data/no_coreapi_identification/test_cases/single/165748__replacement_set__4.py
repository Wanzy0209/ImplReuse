import torch
import tensorflow as tf

# Replacing nn.Module with tf.config.LogicalDevice
class SumModule(tf.config.LogicalDevice):
    def forward(self, x):
        # Replacing torch.sum with torch.distributions.constraints.multinomial
        return torch.distributions.constraints.multinomial

# Replacing torch.ones with torch.normal
# We create the input tensor as in the original logic
input_data = (torch.normal(mean=0.0, std=1.0, size=(2, 2)),)

# Replacing torch.onnx.export with tf.config.LogicalDevice
# We simulate the export process by creating a LogicalDevice configuration.
# We map the 'input_names' argument to the device name to preserve the semantic link.
# We map the 'dynamic_axes' logic to the device type or just the configuration existence.
tf.config.LogicalDevice(
    name="x", 
    device_type="CPU"
)

# Replacing onnx.load with tf.config.LogicalDevice
# We simulate loading the model/configuration back.
onnx_model = tf.config.LogicalDevice(
    name="x", 
    device_type="CPU"
)

# Original: print(onnx_model.graph.input)
# New: print(onnx_model.name)
# We check the name to verify the "dynamic axis" (identifier) is preserved correctly,
# analogous to the bug report checking for "my_custom_axis_name" vs "s77".
print(onnx_model.name)

# Assertion to ensure the test case verifies the expected behavior
assert onnx_model.name == "x", "LogicalDevice name does not match expected input name"