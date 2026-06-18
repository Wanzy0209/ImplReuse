import tensorflow as tf

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