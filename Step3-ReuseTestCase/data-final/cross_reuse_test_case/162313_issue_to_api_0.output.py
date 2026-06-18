import torch
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

# Define custom classes to simulate changing state
class CustomObjectA(tf.keras.layers.Layer):
    pass

class CustomObjectB(tf.keras.layers.Layer):
    pass

# 1. Setup initial global state (analogous to flag = True)
# We use the API to get the reference and modify it
custom_objects = get_custom_objects()
custom_objects['MyObject'] = CustomObjectA

# 2. First access (analogous to fn(torch.ones(3)))
# Verify the initial state is retrieved correctly
retrieved_obj = get_custom_objects()['MyObject']
assert retrieved_obj is CustomObjectA, "Initial state retrieval failed"

# 3. Mutate global state (analogous to flag = False)
# We modify the dictionary directly or via the reference
custom_objects['MyObject'] = CustomObjectB

# 4. Second access (analogous to fn(torch.ones(3)) with flag=False)
# The original bug failed here (KeyError) because internal state wasn't updated.
# We verify that get_custom_objects returns the live, updated reference.
retrieved_obj_updated = get_custom_objects()['MyObject']
assert retrieved_obj_updated is CustomObjectB, "Updated state retrieval failed"

print("Test passed: get_custom_objects correctly reflects live global state changes.")