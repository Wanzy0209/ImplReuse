import torch
import tensorflow as tf

# Disable v2 behavior to use compat.v1 APIs as intended
tf.compat.v1.disable_v2_behavior()

# Define a function that uses the target API and attempts to set a dynamic attribute.
# This mimics the original bug's logic: API call -> Dynamic Attribute -> Return.
def create_producer(limit: int):
    # Call the specific API: tf.compat.v1.train.range_input_producer
    producer = tf.compat.v1.train.range_input_producer(limit)
    
    # Add dynamic attribute (similar to tup.extra_info = ...)
    producer.extra_info = "dynamic_attribute_value"
    
    return producer

print("\nTesting range_input_producer with __setattr__ (Direct):")
# Test 1: Direct call (Eager/Graph construction)
try:
    direct_result = create_producer(10)
    print(f"Direct result extra_info: {direct_result.extra_info}")
except AttributeError as e:
    print(f"Direct result error: {e}")

print("\nTesting range_input_producer with __setattr__ (tf.function):")
# Test 2: Wrapped in tf.function (TensorFlow's equivalent to torch.compile)
# This checks if the compilation/tracing mechanism strips dynamic attributes.
try:
    compiled_fn = tf.function(create_producer)
    compiled_result = compiled_fn(10)
    print(f"Compiled result extra_info: {compiled_result.extra_info}")
except AttributeError as e:
    print(f"Compiled result error: {e}")