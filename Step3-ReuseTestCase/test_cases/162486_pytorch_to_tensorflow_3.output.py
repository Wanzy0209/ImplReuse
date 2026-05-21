import tensorflow as tf

def test_pad_to_cardinality():
    """
    Adapted test case for tf.data.experimental.pad_to_cardinality.
    The original PyTorch test verified splitting a dataset based on proportions.
    This test verifies padding a dataset to a specific cardinality, which is the 
    TensorFlow equivalent for manipulating dataset size/structure.
    """
    
    # Create a dataset with dictionary elements (required by pad_to_cardinality)
    # Original PyTorch test used tensors x (100, 3) and y (100, 2).
    # We use a smaller size for verification clarity.
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    
    # In TensorFlow, we use from_tensor_slices to create the dataset
    dataset = tf.data.Dataset.from_tensor_slices({'x': x, 'y': y})
    
    # Original PyTorch test split into [0.7, 0.2, 0.1].
    # Here we pad the dataset of size 3 to a target cardinality of 5.
    target_cardinality = 5
    
    # Apply the padding operation
    padded_dataset = dataset.apply(
        tf.data.experimental.pad_to_cardinality(target_cardinality)
    )
    
    # Verify the output
    results = list(padded_dataset.as_numpy_iterator())
    
    # Check if the cardinality matches the target
    assert len(results) == target_cardinality, \
        f"Expected cardinality {target_cardinality}, got {len(results)}"
    
    # Verify original data is preserved and marked valid
    for i in range(3):
        assert results[i]['valid'] == True, f"Item {i} should be marked valid"
        assert results[i]['x'] == x[i], f"Item {i} x data mismatch"
        assert results[i]['y'] == y[i], f"Item {i} y data mismatch"
        
    # Verify padding elements are added and marked invalid
    for i in range(3, 5):
        assert results[i]['valid'] == False, f"Item {i} should be marked as padding"
        # Padding defaults to 0 for numeric types
        assert results[i]['x'] == 0, f"Item {i} padding x mismatch"
        assert results[i]['y'] == 0, f"Item {i} padding y mismatch"

    print("Test passed: pad_to_cardinality works as expected.")

if __name__ == "__main__":
    test_pad_to_cardinality()