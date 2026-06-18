import torch
import tensorflow as tf
import tf.experimental.numpy as tnp
import numpy as np

# Mimicking the structure of the original bug report (Model class + Compilation)
class LinspaceTestModel(tf.Module):
    def __init__(self):
        super().__init__()

    # Using tf.function to mimic torch.compile
    @tf.function
    def call(self, start, stop, num, endpoint):
        # The similar API implementation shows specific handling for 'endpoint'
        # and 'num'. We test these branches.
        
        # Branch 1: endpoint=True
        # Implementation: result = math_ops.linspace(start, stop, num, axis=axis)
        res_true = tnp.linspace(start, stop, num=num, endpoint=True)
        
        # Branch 2: endpoint=False
        # Implementation: manual handling with step calculation
        res_false = tnp.linspace(start, stop, num=num, endpoint=False)
        
        return res_true, res_false

def main():
    model = LinspaceTestModel()
    
    # Test parameters
    start = 0.0
    stop = 10.0
    num = 5
    
    # Run the compiled model
    res_true, res_false = model.call(start, stop, num, True)
    
    # Assertions based on the API implementation logic provided
    # For endpoint=True, step = (stop - start) / (num - 1)
    # Expected: [0.0, 2.5, 5.0, 7.5, 10.0]
    expected_true = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    np.testing.assert_allclose(res_true.numpy(), expected_true)
    
    # For endpoint=False, step = (stop - start) / num
    # Expected: [0.0, 2.0, 4.0, 6.0, 8.0]
    expected_false = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(res_false.numpy(), expected_false)
    
    # Test the 'num < 0' validation mentioned in the API code
    try:
        tnp.linspace(0, 10, -1)
        assert False, "Expected ValueError for num < 0"
    except ValueError as e:
        assert "non-negative" in str(e)
        
    print("Test passed.")

if __name__ == "__main__":
    main()