import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2

# Setup: Create dummy TensorInfo objects to serve as inputs and outputs
# This mimics the setup phase in the PyTorch example (creating the Linear layer)
input_tensor_info = meta_graph_pb2.TensorInfo()
input_tensor_info.name = "input_tensor:0"
# Fix: Use tf.float.as_datatype_enum to get the correct enum value for the protobuf
input_tensor_info.dtype = tf.float.as_datatype_enum

output_tensor_info = meta_graph_pb2.TensorInfo()
output_tensor_info.name = "output_tensor:0"
# Fix: Use tf.float.as_datatype_enum to get the correct enum value for the protobuf
output_tensor_info.dtype = tf.float.as_datatype_enum

inputs = {"x": input_tensor_info}
outputs = {"y": output_tensor_info}

# Call the target API: build_signature_def
# This mimics the prune.remove call in the original example
signature_def = tf.compat.v1.saved_model.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name="predict_signature"
)

# Verification: Inspect the result
# This mimics the final 'm' print in the original example
print(signature_def)

# Basic assertions to ensure the API behaved as expected
assert signature_def.method_name == "predict_signature"
assert "x" in signature_def.inputs
assert "y" in signature_def.outputs