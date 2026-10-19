import tensorflow as tf
import torch

# Original logic: version.parse(torch.version.hip)
# Replaced logic: torch.backends.cuda.is_built()
# We replace the version parsing check with a backend availability check.
# The similar API tf.keras.backend.floatx returns a string, while this replacement returns a bool.
# We verify the return type to ensure the API is functioning correctly,
# similar to how version.parse verifies the version string format.

is_built = torch.backends.cuda.is_built()
assert isinstance(is_built, bool), "torch.backends.cuda.is_built should return a boolean"
print(f"Backend built status: {is_built}")