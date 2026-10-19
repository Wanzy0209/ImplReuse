import sys
import platform
import torch
import tensorflow as tf

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("TensorFlow:", tf.__version__)
print("macOS:", platform.mac_ver()[0])
print("platform:", platform.platform())
print("arch:", platform.machine())
print("is_macos_or_newer(14,0):",
      getattr(torch.backends.mps, "is_macos_or_newer", lambda *a: "no-func")(14,0))

# Replacement for torch.backends.mps.is_built
# Translating semantics to check availability of the TensorFlow Profiler Trace API
print("TF Profiler Trace available?:",
      hasattr(tf.profiler.experimental, 'client') and hasattr(tf.profiler.experimental.client, 'trace'))

# Original API (torch.backends.mps.is_available)
print("MPS available?:", torch.backends.mps.is_available())