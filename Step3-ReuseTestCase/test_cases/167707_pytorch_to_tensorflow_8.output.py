import tensorflow as tf

# Define a custom class to represent the object being handled (mimicking the trace data)
class CustomTraceHandler:
    pass

# Setup configuration
# Mimics the 'worker_name' argument in the original bug report
object_name = "trace"
# Mimics the handler configuration passed to the profiler
custom_objects = {object_name: CustomTraceHandler}

# Action: Retrieve the object using the similar API
# This corresponds to the logic where a handler or component is looked up by name
retrieved_obj = tf.keras.utils.get_registered_object(
    object_name,
    custom_objects=custom_objects
)

# Verification: Ensure the object is retrieved correctly and is not "broken"
# The original bug resulted in a broken trace file; here we verify the lookup integrity
assert retrieved_obj is not None, "Retrieved object is None (lookup failed)"
assert retrieved_obj is CustomTraceHandler, \
    f"Expected {CustomTraceHandler}, but got {retrieved_obj}"