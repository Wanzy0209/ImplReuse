import tensorflow as tf

def test_varlen_feature_handles_missing_data():
    """
    This test case is derived from the logic of Issue 160150, where the PyTorch ONNX 
    exporter crashed when a node output included None.
    
    Here, we test the similar API `tf.io.VarLenFeature` to ensure it handles 
    missing or 'None-like' data gracefully. In the context of TensorFlow parsing, 
    a missing feature is analogous to a None value. We verify that the parser 
    returns an empty SparseTensor rather than crashing or returning a raw None.
    """
    
    # Define a feature configuration using the similar API: tf.io.VarLenFeature
    # This is used to parse data that might be missing or variable length.
    feature_spec = {
        'image_tokens_masks': tf.io.VarLenFeature(tf.float32)
    }

    # Create a serialized example where 'image_tokens_masks' is MISSING.
    # This simulates the scenario in the bug report: "image_tokens_masks could be None"
    example = tf.train.Example(features=tf.train.Features(feature={
        'output': tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3]))
    }))
    serialized = example.SerializeToString()

    # Parse the example
    # In the original bug, the exporter crashed here when encountering None.
    # We expect tf.io to handle the missing key gracefully.
    parsed_features = tf.io.parse_single_example(serialized, feature_spec)

    # Assertions to verify correct behavior
    # 1. The key should exist in the output dictionary
    assert 'image_tokens_masks' in parsed_features
    
    # 2. The value should be a SparseTensor (the TF representation of variable/missing data)
    assert isinstance(parsed_features['image_tokens_masks'], tf.SparseTensor)
    
    # 3. The SparseTensor should be empty (indices and values are empty tensors)
    # This confirms the API handled the "None" input correctly without crashing.
    assert parsed_features['image_tokens_masks'].indices.shape[0] == 0
    assert parsed_features['image_tokens_masks'].values.shape[0] == 0

if __name__ == "__main__":
    test_varlen_feature_handles_missing_data()
    print("Test passed: VarLenFeature correctly handled missing data.")