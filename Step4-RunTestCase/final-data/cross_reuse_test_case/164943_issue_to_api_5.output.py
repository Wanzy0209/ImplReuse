import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

# Test case for tf.feature_column.crossed_column
# This test mirrors the validation logic error found in the PyTorch MPS backend issue.
# The PyTorch issue involved a version check raising a RuntimeError.
# Here we test the argument validation (hash_bucket_size) of crossed_column.

def test_crossed_column_validation():
    # Test with invalid hash_bucket_size (should raise ValueError)
    # This mirrors the "Error" condition in the original bug report.
    try:
        fc.crossed_column(['key1', 'key2'], -1)
        assert False, "Expected ValueError for negative hash_bucket_size"
    except ValueError as e:
        assert "hash_bucket_size" in str(e)

    # Test with valid inputs (should succeed)
    # This mirrors the expected behavior of the PyTorch code.
    col = fc.crossed_column(['key1', 'key2'], 100)
    
    # Fix: The internal class _CrossedColumn is not exposed in the module attribute list in some TF versions.
    # We check against the public base class CategoricalColumn instead.
    assert isinstance(col, tf.feature_column.CategoricalColumn)

if __name__ == "__main__":
    test_crossed_column_validation()
    print("Test passed.")