#!/usr/bin/env python3

"""
Script to test tf.io.VarLenFeature parsing correctness.
Focuses on verifying that the parsed SparseTensor shapes match the input data,
analogous to verifying ONNX tensor shapes in the original bug report.
"""

import tensorflow as tf
import numpy as np


def create_feature_spec(dtype=tf.float32):
    """
    Create a feature specification using tf.io.VarLenFeature.
    
    Returns:
        dict: A feature specification dictionary.
    """
    feature_spec = {
        'varlen_data': tf.io.VarLenFeature(dtype=dtype)
    }
    print(f"Created feature specification with dtype: {dtype}")
    return feature_spec


def parse_and_verify(feature_spec, serialized_data, expected_values):
    """
    Parse the serialized data and verify the output tensor properties.
    
    Args:
        feature_spec (dict): The feature specification.
        serialized_data (str): Serialized tf.train.Example string.
        expected_values (list): The expected values in the sparse tensor.
    """
    # Parse the example
    parsed_features = tf.io.parse_single_example(serialized_data, feature_spec)
    
    # Extract the sparse tensor
    sparse_tensor = parsed_features['varlen_data']
    
    # Verify the tensor is a SparseTensor
    assert isinstance(sparse_tensor, tf.SparseTensor), "Parsed feature should be a SparseTensor"
    
    # Verify values
    np.testing.assert_array_equal(sparse_tensor.values.numpy(), expected_values)
    
    # Verify shape
    # The original bug reported incorrect bias shapes. Here we verify the dense_shape
    # of the sparse tensor correctly reflects the number of elements.
    expected_shape = np.array([len(expected_values)], dtype=np.int64)
    np.testing.assert_array_equal(sparse_tensor.dense_shape.numpy(), expected_shape)
    
    print(f"Verification successful. Shape: {sparse_tensor.dense_shape.numpy()}, Values: {sparse_tensor.values.numpy()}")


def main():
    """Main function to create spec, serialize data, and verify parsing."""
    
    print("Creating feature specification...")
    feature_spec = create_feature_spec()
    
    # Create dummy data
    # Analogous to dummy_input in the original script
    data_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Serialize data to tf.train.Example
    print("Serializing example data...")
    example = tf.train.Example(features=tf.train.Features(feature={
        'varlen_data': tf.train.Feature(float_list=tf.train.FloatList(value=data_values))
    }))
    serialized = example.SerializeToString()
    
    # Parse and verify
    print("Parsing and verifying...")
    parse_and_verify(feature_spec, serialized, data_values)
    
    print("Test completed successfully.")


if __name__ == "__main__":
    main()