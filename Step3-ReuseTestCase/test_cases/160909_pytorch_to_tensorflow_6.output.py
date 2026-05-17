import torch
import tensorflow as tf

# Define a simple model (equivalent to Model() in the original PyTorch code)
class SimpleModel(tf.Module):
    def __call__(self, x):
        # Mimicking a simple operation similar to the repeat_interleave/add logic
        return x + 1

def test_name_scope():
    # Initialize model
    model = SimpleModel()
    
    # Use the similar API: tf.compat.v1.name_scope
    # This acts as the wrapper/context similar to torch.compile in the original snippet.
    # We define a scope name to group operations.
    with tf.compat.v1.name_scope("my_custom_scope"):
        # Create data (equivalent to data.to('PrivateUse1'))
        # Note: tf.compat.v1.name_scope handles naming, not device placement directly,
        # but we verify the context is applied correctly.
        data = tf.constant([1.0, 2.0, 3.0])
        
        # Run the model (equivalent to compiled(data))
        result = model(data)

    # Verify the behavior: Check if the operation was created within the scope.
    # In the original bug, meta tensors appeared on the wrong device.
    # Here, we verify that the scope context is correctly applied to the operations.
    assert "my_custom_scope" in result.op.name, \
        f"Operation name '{result.op.name}' does not contain scope 'my_custom_scope'. " \
        "This indicates the context manager did not wrap the operations as expected."
    
    # Verify the result is correct
    expected = [2.0, 3.0, 4.0]
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Result values are incorrect"

    print("Test passed: name_scope correctly wrapped the model operations.")

if __name__ == "__main__":
    test_name_scope()