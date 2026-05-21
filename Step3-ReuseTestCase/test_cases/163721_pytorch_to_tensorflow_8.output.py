import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# Enable numpy behavior for tensorflow.experimental.numpy
tnp.experimental_enable_numpy_behavior()

# Check if TensorFlow is available (mimicking the MPS availability check)
assert tf.__version__ is not None

# Wrapper over the custom iscomplex logic, mimicking the structure of MPSSoftshrink
class TFIsComplexWrapper(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        # Using the similar API: tf.experimental.numpy.iscomplex
        return tnp.iscomplex(input)

# Wrapper over a model, mimicking CustomMPSSoftshrinkModel
# Note: Since iscomplex returns booleans, we adapt the model to simply apply the check
# rather than chaining Linear layers which expect float inputs.
class CustomTFModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.wrapper = TFIsComplexWrapper()

    def __call__(self, x):
        return self.wrapper(x)

# Test execution
if __name__ == "__main__":
    model = CustomTFModel()

    # Test Case 1: Real numbers
    real_input = tf.constant([1.0, 2.0, 3.0])
    output_real = model(real_input)
    # Verify behavior: Real numbers should return False
    assert tf.reduce_all(output_real == False).numpy(), "Real numbers should not be complex"

    # Test Case 2: Complex numbers
    complex_input = tf.constant([1.0 + 2.0j, 3.0 + 4.0j])
    output_complex = model(complex_input)
    # Verify behavior: Complex numbers should return True
    assert tf.reduce_all(output_complex == True).numpy(), "Complex numbers should be complex"

    print("Test passed.")