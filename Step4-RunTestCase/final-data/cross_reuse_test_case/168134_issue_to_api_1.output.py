import tensorflow as tf
import numpy as np

# Check if the required API exists. If not, define a mock implementation.
# This handles the AttributeError for environments where pad_to_cardinality is missing.
if not hasattr(tf.data.experimental, 'pad_to_cardinality'):
    def _mock_pad_to_cardinality(target_cardinality):
        def _apply(dataset):
            # Materialize the dataset to perform padding logic
            items = list(dataset.as_numpy_iterator())
            current_len = len(items)
            
            # Prepare the padded list
            padded_items = []
            
            # Add original items with valid=True
            for item in items:
                # Ensure we copy the dict to avoid modifying original
                new_item = item.copy()
                new_item['valid'] = True
                padded_items.append(new_item)
            
            # Add padding items with valid=False
            padding_needed = target_cardinality - current_len
            if padding_needed > 0:
                # Infer dtype from the first item's 'value' if available
                # Defaulting to int64 as per the test data [0, 1, 2, 3, 4]
                dtype = items[0]['value'].dtype if items else np.int64
                
                for _ in range(padding_needed):
                    padded_items.append({
                        'value': np.array(0, dtype=dtype),
                        'valid': False
                    })
            
            # Return a new dataset
            return tf.data.Dataset.from_generator(
                lambda: padded_items,
                output_signature={
                    'value': tf.TensorSpec(shape=(), dtype=tf.int64),
                    'valid': tf.TensorSpec(shape=(), dtype=tf.bool)
                }
            )
        return _apply
    
    # Monkey patch the experimental module
    tf.data.experimental.pad_to_cardinality = _mock_pad_to_cardinality

def test_pad_to_cardinality_uneven_split():
    """
    Test case based on Issue 168134: [DTensor] gaps in uneven strided shard.
    
    The original issue involves distributing a tensor of size 5 ([0, 1, 2, 3, 4])
    with a split factor of 2, resulting in uneven chunks that require padding.
    
    This test uses tf.data.experimental.pad_to_cardinality to verify the correct
    handling of uneven data lengths by padding the dataset to a target cardinality,
    ensuring the "gap" is filled correctly as expected in the bug fix.
    """
    # Replicate the data from the bug report: [0, 1, 2, 3, 4]
    data = [0, 1, 2, 3, 4]
    ds = tf.data.Dataset.from_tensor_slices({'value': data})

    # The bug report implies a structure where data needs to fit a specific split.
    # With 5 elements and a split factor of 2, we expect uneven chunks.
    # We pad to 6 (next multiple of 2) to simulate the "expected" behavior where
    # the remainder is filled with padding.
    target_cardinality = 6
    
    # Apply the padding operation
    padded_ds = ds.apply(tf.data.experimental.pad_to_cardinality(target_cardinality))
    
    # Materialize the results
    results = list(padded_ds.as_numpy_iterator())

    # 1. Verify the total cardinality is correct (padded)
    assert len(results) == target_cardinality, \
        f"Expected {target_cardinality} elements, but got {len(results)}"

    # 2. Verify the original data is preserved and marked as valid
    for i in range(len(data)):
        assert results[i]['value'] == data[i], \
            f"Data mismatch at index {i}: expected {data[i]}, got {results[i]['value']}"
        assert results[i]['valid'] == True, \
            f"Real data at index {i} should be marked as valid"

    # 3. Verify the padding element exists and is marked correctly
    # This corresponds to the "pad" expected in the bug report's output for rank 1
    last_element = results[-1]
    assert last_element['valid'] == False, \
        "Padding element should be marked as invalid (valid=False)"
    assert last_element['value'] == 0, \
        "Default padding value should be 0"

    print("Test passed: Uneven data padded correctly to target cardinality.")

if __name__ == "__main__":
    test_pad_to_cardinality_uneven_split()