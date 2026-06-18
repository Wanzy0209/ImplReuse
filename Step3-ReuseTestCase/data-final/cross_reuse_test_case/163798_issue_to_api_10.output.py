import torch
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

# Define a custom type to mimic the context of the bug where a tensor-like object
# interacts with scalar values.
@tf.experimental.ExtensionType
class CustomTensor:
    values: tf.Tensor

# Use the similar API (tf.experimental.dispatch_for_binary_elementwise_apis)
# to handle binary operations (like mul) for the CustomTensor.
# This relates to the bug report which focuses on the behavior of 
# multiplication (a*u0*u1) involving tensor and scalar interactions.
@tf.experimental.dispatch_for_binary_elementwise_apis(CustomTensor, (int, float, tf.Tensor))
def custom_tensor_binary_op(api_func, x, y):
    # x is CustomTensor, y is the scalar/tensor (e.g., u0 or u1 from the bug)
    # Delegate to the standard api_func (e.g. tf.multiply) on the wrapped values
    return api_func(x.values, y)

# Handle the reverse case (scalar * CustomTensor) to support a*u0*u1 associativity
@tf.experimental.dispatch_for_binary_elementwise_apis((int, float, tf.Tensor), CustomTensor)
def custom_tensor_binary_op_reverse(api_func, x, y):
    return api_func(x, y.values)

class TestDispatchForBinaryOps(test_util.TensorFlowTestCase):
    def test_multiplication_with_scalars_from_list(self):
        """
        Test case derived from Issue 163798.
        
        The original bug involves the graphing behavior of:
            u0, u1 = a.tolist()
            return a*u0*u1
            
        This test adapts that logic to TensorFlow using the similar API.
        We verify that the binary elementwise dispatch correctly handles
        the multiplication of a custom tensor type with scalar values
        (simulating the unpacking from tolist()).
        """
        # Create a custom tensor 'a' analogous to torch.tensor([1, 2])
        a = CustomTensor(tf.constant([1, 2]))
        
        # Simulate u0, u1 = a.tolist()
        # In the bug, these are Python scalars extracted from the tensor.
        u0 = 2
        u1 = 3
        
        # Perform the operation sequence from the bug: return a*u0*u1
        # This relies on the dispatch_for_binary_elementwise_apis handler
        # to correctly bridge the CustomTensor with the scalar integers.
        result = a * u0 * u1
        
        # Expected: [1, 2] * 2 * 3 = [6, 12]
        expected = tf.constant([6, 12])
        
        # Assert the dispatch worked correctly and the arithmetic matches the bug's logic
        self.assertAllEqual(result, expected)

if __name__ == "__main__":
    test.main()