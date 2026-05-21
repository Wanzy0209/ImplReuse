import sys
import platform
import tensorflow as tf

print("Python:", sys.version)
print("Executable:", sys.executable)
print("TensorFlow:", tf.__version__)
print("OS:", platform.platform())
print("Arch:", platform.machine())

# Check if the API exists (analogous to is_built)
is_built = hasattr(tf.experimental, 'enable_strict_mode')
print("Strict mode built?:", is_built)

# Check if the mode can be enabled (analogous to is_available)
is_available = False
if is_built:
    try:
        # Attempt to enable strict mode
        tf.experimental.enable_strict_mode()
        is_available = True
    except Exception as e:
        # If enabling fails, it is considered not available/working
        print(f"Error enabling strict mode: {e}")

print("Strict mode available?:", is_available)