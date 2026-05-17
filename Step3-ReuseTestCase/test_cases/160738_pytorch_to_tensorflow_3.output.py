import tensorflow as tf
import tf.experimental.numpy as tnp

def test_nextafter(device_name):
    # Create zero-dimensional tensors (scalars) on the specified device
    with tf.device(device_name):
        x1 = tnp.array(3.0)
        x2 = tnp.array(4.0)
        try:
            # Note: nextafter is a binary element-wise operation and does not take a 'dim' argument.
            # We test the handling of zero-dimensional inputs here.
            output = tnp.nextafter(x1, x2)
            print(f"nextafter test succeeds for device: {device_name}. output: {output}")
        except Exception as e:
            print(f"nextafter test fails for device: {device_name}: {e}")

# Test on CPU
test_nextafter("/cpu:0")

# Test on GPU if available, otherwise skip
try:
    test_nextafter("/gpu:0")
except Exception as e:
    # Ignore if GPU is not found or other device placement errors
    pass