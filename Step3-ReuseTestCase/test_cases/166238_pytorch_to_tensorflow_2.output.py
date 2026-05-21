import torch
import unittest
import collections
import tensorflow as tf
from tensorflow.experimental import dtensor

class TestDTensorDefaultDictInteraction(unittest.TestCase):
    """
    Adapted test case based on PyTorch Dynamo Issue #166238.
    
    Original Issue: torch.compile (Dynamo) failed to trace the creation of 
    collections.defaultdict, resulting in a graph break/Unsupported error.
    
    This test verifies the behavior of the similar TensorFlow API 
    (tf.experimental.dtensor.copy_to_mesh) when used within a traced context 
    (tf.function) that also instantiates a collections.defaultdict.
    """

    def test_copy_to_mesh_with_defaultdict_in_trace(self):
        """
        Tests if copy_to_mesh works correctly when a defaultdict is created
        inside the same tf.function (tracing context).
        """
        # Setup: Create a mesh and layout required for copy_to_mesh
        # Using CPU device for general compatibility in test environments
        mesh = dtensor.create_mesh(['batch'], ['device:CPU:0'])
        layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

        # The function to be traced (equivalent to the function passed to torch.compile)
        @tf.function
        def func_with_defaultdict_and_copy(tensor):
            # Core bug reproduction logic:
            # PyTorch Dynamo failed here: "Dynamo does not know how to trace 
            # the function `<class 'collections.defaultdict'>`"
            dd = collections.defaultdict(list)
            dd['logs'].append('start')
            
            # Call the Similar API: copy_to_mesh
            # We verify if this operation succeeds despite the defaultdict creation
            result_tensor = dtensor.copy_to_mesh(tensor, layout)
            
            dd['logs'].append('end')
            return result_tensor

        # Input data
        input_tensor = tf.constant([1.0, 2.0, 3.0])

        # Execution
        # In the original PyTorch bug, this raised:
        # torch._dynamo.exc.Unsupported: Unsupported function call ...
        try:
            result = func_with_defaultdict_and_copy(input_tensor)
            
            # Assertions
            # 1. Verify the API returned the expected type (DTensor)
            self.assertIsInstance(result, dtensor.DTensor)
            
            # 2. Verify data integrity
            self.assertAllEqual(result.values, input_tensor)
            
        except Exception as e:
            self.fail(f"tf.function (tracing) failed to handle defaultdict with copy_to_mesh. Error: {e}")

if __name__ == '__main__':
    unittest.main()