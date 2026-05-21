import torch
import tensorflow as tf

def test_output_all_intermediates_int_conversion():
    """
    Test case based on PyTorch Issue #164297.
    
    The original bug involved a segmentation fault when accessing the __int__ 
    method of a C-bound object (torch.onnx.OperatorExportTypes) during import.
    
    This test verifies the similar TensorFlow API (tf.compat.v1.experimental.output_all_intermediates)
    to ensure that accessing the return value and converting it to an integer
    (a boolean in this case) does not cause memory access violations or crashes.
    """
    # Access the API which returns a configuration boolean
    result = tf.compat.v1.experimental.output_all_intermediates()
    
    # The PyTorch crash occurred specifically in the dispatcher for __int__.
    # We explicitly test the integer conversion here to ensure the binding is safe.
    int_value = int(result)
    
    # Assertions to verify valid behavior
    assert isinstance(result, bool), "API should return a boolean"
    assert int_value in [0, 1], "Integer conversion of boolean should be 0 or 1"

if __name__ == "__main__":
    test_output_all_intermediates_int_conversion()
    print("Test passed successfully.")