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

# Check macOS version helper
print("is_macos_or_newer(14,0):",
      getattr(torch.backends.mps, "is_macos_or_newer", lambda *a: "no-func")(14, 0))

# Replacement for torch.backends.mps.is_built
# Original: print("MPS built?:", torch.backends.mps.is_built())
# Replacement: tf.compat.v1.linalg.trace
# We create a dummy tensor to compute the trace.
dummy_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
trace_val = tf.compat.v1.linalg.trace(dummy_tensor)
print("TF Trace result (replacing is_built):", trace_val)

# Original API (Same as Similar API)
print("MPS available?:", torch.backends.mps.is_available())