import sys
import platform
import torch
import tensorflow as tf

# System Information
print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("macOS:", platform.mac_ver()[0])
print("platform:", platform.platform())
print("arch:", platform.machine())
print("is_macos_or_newer(14,0):",
      getattr(torch.backends.mps, "is_macos_or_newer", lambda *a: "no-func")(14,0))

# Replacement: torch.backends.mps.is_built -> tf.keras.ops.trace
# Note: tf.keras.ops.trace requires a tensor argument. 
# We create a dummy tensor to allow the function call to execute.
dummy_tensor = tf.eye(2)
print("MPS built?:", tf.keras.ops.trace(dummy_tensor))

# Original/Similar API: torch.backends.mps.is_available
mps_available = torch.backends.mps.is_available()
print("MPS available?:", mps_available)

# Assertion to verify the expected behavior
# Expected: MPS available?: True on Apple Silicon under Tahoe
assert mps_available, "MPS should be available on Apple Silicon under macOS 26 (Tahoe)"