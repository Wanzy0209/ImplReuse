import torch
import sys
import platform
import tensorflow as tf

print("Python:", sys.version)
print("Executable:", sys.executable)
print("TensorFlow:", tf.__version__)
print("OS:", platform.platform())
print("Arch:", platform.machine())

# Check for TPU availability (analogous to torch.backends.mps.is_available)
try:
    tpu_devices = tf.config.list_physical_devices('TPU')
    print("TPU available?:", len(tpu_devices) > 0)
except Exception as e:
    print("TPU available?: False (Error: {})".format(e))

# Test the specific API: tf.compat.v1.tpu.core
# This API generates the device name for a TPU core.
try:
    core_num = 0
    device_name = tf.compat.v1.tpu.core(core_num)
    print("TPU core({}) result: {}".format(core_num, device_name))
    
    # Verify the expected format
    expected_name = "device:TPU_REPLICATED_CORE:{}".format(core_num)
    assert device_name == expected_name, "Unexpected device name format"
    print("Test Passed: tf.compat.v1.tpu.core returned expected format.")
except AttributeError:
    print("Test Failed: tf.compat.v1.tpu.core is not available (API not built).")
except Exception as e:
    print("Test Failed with exception:", e)