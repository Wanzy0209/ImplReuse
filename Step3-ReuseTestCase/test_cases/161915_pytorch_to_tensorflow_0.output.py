import torch
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate

def test_get_source_inputs():
    """
    Adapted test case for tf.keras.utils.get_source_inputs based on 
    PyTorch NestedTensor share_memory_() issue.
    
    Original Logic:
    1. Create tensors (a, b).
    2. Create a nested structure (nt).
    3. Call a method on the structure (share_memory_).
    
    Adapted Logic:
    1. Create symbolic input tensors (a, b).
    2. Create a combined structure (merged_tensor).
    3. Call the utility function on the structure (get_source_inputs).
    """
    
    # 1. Create inputs (mimicking torch.randn(3) and torch.randn(5))
    # In Keras, we use Input layers to establish the graph history required by get_source_inputs
    a = Input(shape=(3,), name="input_a")
    b = Input(shape=(5,), name="input_b")

    # 2. Create a structure (mimicking torch.nested.nested_tensor([a, b]))
    # We combine the inputs into a single tensor structure using Concatenate
    merged_tensor = Concatenate()([a, b])

    # 3. Call the API (mimicking nt.share_memory_())
    # The original bug resulted in a Segmentation Fault.
    # Here we verify that get_source_inputs handles the structure correctly.
    try:
        source_inputs = tf.keras.utils.get_source_inputs(merged_tensor)
        
        # Assertions to verify correct behavior
        assert source_inputs is not None, "get_source_inputs returned None"
        assert isinstance(source_inputs, list), "get_source_inputs did not return a list"
        assert len(source_inputs) == 2, f"Expected 2 source inputs, got {len(source_inputs)}"
        
        print("Test Passed: get_source_inputs successfully retrieved source inputs.")
        print(f"Source inputs: {source_inputs}")
        
    except Exception as e:
        print(f"Test Failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_get_source_inputs()