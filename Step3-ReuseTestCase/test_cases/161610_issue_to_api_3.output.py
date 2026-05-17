import tensorflow as tf
import os
import tempfile

# Define a class structure analogous to the NamedTuple subclass in the bug report
# In TensorFlow, tf.Module is the standard base class for savable objects
class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        # Initialize some variables to make it a valid model
        self.first = tf.Variable([2.0])
        self.second = tf.Variable([1.0])

    # Including a tf.function method to align with the similar API usage pattern
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add(self, x):
        return x + 1.0

def test_dynamic_attribute_persistence():
    print("\nTesting tf.saved_model.save with dynamic attributes:")
    
    # Create an instance
    module = MyModule()
    
    # Add a dynamic attribute (mimicking the bug report's 'extra_info')
    extra_info = 4.0
    module.extra_info = extra_info
    
    # Save the model using the similar API
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "saved_model")
        tf.saved_model.save(module, export_path)
        
        # Load the model
        loaded_module = tf.saved_model.load(export_path)
        
        # Check if the dynamic attribute persists
        # This mirrors the check in the PyTorch bug report
        try:
            print(f"tf.saved_model.load result: {loaded_module.extra_info}")
        except AttributeError as e:
            print(f"AttributeError: {e}")

if __name__ == "__main__":
    test_dynamic_attribute_persistence()