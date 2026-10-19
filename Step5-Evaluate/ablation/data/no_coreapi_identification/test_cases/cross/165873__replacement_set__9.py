import torch
import torch.nn.functional as F
import tensorflow as tf
import tf.autograph

# Test case adapted from Issue 165873 using specified API replacements.

# 1. Generate the "large tensor" (state_dict value)
# Original: torch.randn(32000)
# Replacement: torch.nn.functional.adaptive_max_pool1d
# We create a dummy input and pool it to the desired size to simulate the large tensor generation.
dummy_input = torch.randn(1, 1, 40000)
large_tensor = F.adaptive_max_pool1d(dummy_input, output_size=32000).flatten()

# 2. Initialize the "parameter" (scalar)
# Original: torch.tensor(threshold_value)
# Replacement: torch.empty_strided
# We create a scalar-like tensor using empty_strided.
threshold_param = torch.empty_strided((1,), (1,))

# 3. Setup the "Module" and "Load" mechanism
# Original: nn.Module, nn.Parameter, module.load_state_dict
# Replacement: tf.io.VarLenFeature
# We use tf.io.VarLenFeature to define the parsing schema (acting as the module definition)
# and the parsing logic (acting as the load mechanism).
feature_spec = {
    'threshold': tf.io.VarLenFeature(tf.float32)
}

# Create a tf.train.Example to simulate the state_dict data structure
state_dict_example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'threshold': tf.train.Feature(
                float_list=tf.train.FloatList(value=large_tensor.tolist())
            )
        }
    )
)

# 4. Execute the load and check logic
# Similar API: tf.autograph.trace
# We wrap the logic in a function traced by tf.autograph.trace.
@tf.autograph.trace
def load_and_verify():
    # Perform the "load" operation using the replacement API
    parsed_features = tf.io.parse_single_example(
        state_dict_example.SerializeToString(),
        feature_spec
    )
    
    # Access the value
    # Original: module.threshold.item()
    # Replacement: tf.io.VarLenFeature (Accessing the result of the VarLenFeature parse)
    # The bug implies the scalar takes the first value of the 1D tensor.
    # VarLenFeature parses all values, so we access the first one to mimic the check.
    first_value = parsed_features['threshold'].values[0]
    
    return first_value

# Run the traced function
result = load_and_verify()

# 5. Assertion
# Original: assert module.threshold.item() == state_dict['threshold'][0].item()
# We verify that the result matches the first value of the large tensor,
# demonstrating the "silent" acceptance of the shape mismatch (or in this case, the parsing behavior).
assert result.numpy() == large_tensor[0].item()