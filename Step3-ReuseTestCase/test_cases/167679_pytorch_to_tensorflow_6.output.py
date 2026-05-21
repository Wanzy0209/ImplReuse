import sys, platform, tensorflow as tf

print("Python:", sys.version)
print("Executable:", sys.executable)
print("TensorFlow:", tf.__version__)
print("Platform:", platform.platform())
print("Arch:", platform.machine())

# Check for GPU devices (Analogous to checking if MPS is built/available)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", len(gpus))

# Check the specific API
try:
    # Note: TF32 is enabled by default on supported hardware (Ampere+)
    is_enabled = tf.config.experimental.tensor_float_32_execution_enabled()
    print("TF32 enabled?:", is_enabled)
    
    # Assertion to verify the API behaves as expected (returns a boolean)
    assert isinstance(is_enabled, bool), "Expected boolean return value"
except Exception as e:
    print(f"Error checking TF32 status: {e}")