import torch
import tensorflow as tf
import numpy as np

def test_varlen_feature_no_silent_skip():
    """
    Test that tf.io.VarLenFeature does not silently skip parsing
    when used in a compiled context (tf.function), similar to the
    torch.compile + DebugMode issue where compilation was silently skipped.
    """
    # 1. Setup data with variable length
    feature_data = [1, 2, 3, 4]
    example = tf.train.Example(features=tf.train.Features(feature={
        'var_len_feature': tf.train.Feature(int64_list=tf.train.Int64List(value=feature_data))
    }))
    serialized = example.SerializeToString()

    # 2. Define parsing configuration using the Similar API (tf.io.VarLenFeature)
    feature_spec = {
        'var_len_feature': tf.io.VarLenFeature(tf.int64)
    }

    # 3. Define a parsing function wrapped in tf.function (mimicking torch.compile)
    # This checks if the feature works correctly in a "compiled" mode.
    @tf.function
    def parse_example(serialized_str):
        return tf.io.parse_single_example(serialized_str, feature_spec)

    # 4. Execute
    parsed = parse_example(serialized)

    # 5. Assertions to ensure no silent skip/failure
    # The feature should exist and be a SparseTensor
    assert 'var_len_feature' in parsed, "VarLenFeature was silently skipped/missing"
    assert isinstance(parsed['var_len_feature'], tf.SparseTensor), "VarLenFeature did not return a SparseTensor"

    # The values should match the input
    result_values = parsed['var_len_feature'].values.numpy()
    np.testing.assert_array_equal(result_values, feature_data)

    # Test with empty input to ensure it returns empty SparseTensor, not None or skipped
    empty_example = tf.train.Example(features=tf.train.Features(feature={
        'var_len_feature': tf.train.Feature(int64_list=tf.train.Int64List(value=[]))
    }))
    empty_serialized = empty_example.SerializeToString()
    parsed_empty = parse_example(empty_serialized)

    assert isinstance(parsed_empty['var_len_feature'], tf.SparseTensor), "Empty VarLenFeature failed to return SparseTensor"
    assert len(parsed_empty['var_len_feature'].values.numpy()) == 0, "Empty VarLenFeature has incorrect values"

if __name__ == "__main__":
    test_varlen_feature_no_silent_skip()
    print("Test passed.")