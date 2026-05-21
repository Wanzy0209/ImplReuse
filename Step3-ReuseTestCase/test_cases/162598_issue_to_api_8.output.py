import tensorflow as tf

def test_varlen_feature_missing_data():
    """
    Test case for tf.io.VarLenFeature based on Issue #162598.
    
    The issue describes a CI failure where a download step found 0 objects
    (missing artifact) and subsequently failed when trying to unzip the
    non-existent file.
    
    tf.io.VarLenFeature is the TensorFlow API designed to handle features that
    may be missing or have variable lengths in the input data. This test
    verifies that VarLenFeature correctly handles a "missing" feature
    (analogous to the "0 objects found" in the bug report) by returning an
    empty SparseTensor rather than raising an error, thus demonstrating the
    robust handling of missing data.
    """
    
    # Define the parsing configuration.
    # We use VarLenFeature for a key that might be missing from the input.
    feature_spec = {
        'optional_feature': tf.io.VarLenFeature(tf.float32),
        'required_feature': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    }

    # Create a serialized example where 'optional_feature' is MISSING.
    # This simulates the "Found 0 objects" state from the bug report.
    example = tf.train.Example(features=tf.train.Features(feature={
        'required_feature': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0]))
        # 'optional_feature' is intentionally omitted to simulate the missing artifact
    }))
    
    serialized_example = example.SerializeToString()

    # Parse the example
    parsed = tf.io.parse_single_example(serialized_example, feature_spec)

    # Retrieve the result for the missing feature
    result_tensor = parsed['optional_feature']

    # Assertions
    # 1. The result should be a SparseTensor
    assert isinstance(result_tensor, tf.SparseTensor), \
        "VarLenFeature should return a SparseTensor"
    
    # 2. The tensor should be empty (indices and values length 0).
    # This corresponds to the "0 objects found" log in the issue.
    assert tf.equal(tf.size(result_tensor.indices), 0).numpy(), \
        "Indices should be empty for missing feature"
    assert tf.equal(tf.size(result_tensor.values), 0).numpy(), \
        "Values should be empty for missing feature"

if __name__ == "__main__":
    test_varlen_feature_missing_data()
    print("Test passed: VarLenFeature correctly handled missing data.")