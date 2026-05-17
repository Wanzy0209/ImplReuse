import torch
import tensorflow as tf
import tempfile
import os

def test_text_file_index_type_handling():
    """
    Test case for tf.lookup.TextFileIndex adapted from the logic of the 
    PyTorch float64 Matmul bug report.
    
    The original bug involved setting a specific default data type (float64),
    creating data, performing an operation, and accessing a result.
    
    This test mirrors that structure by:
    1. Defining specific data types (int64 for keys, string for values) via TextFileIndex.
    2. Creating a data source (temporary text file).
    3. Performing the initialization operation (TextFileInitializer).
    4. Accessing a specific result to verify correctness.
    """
    
    # Setup: Create a temporary file with data
    # Mirrors: x = torch.randn(1000, 1000, device="cuda")
    data_content = "alpha\nbeta\ngamma\ndelta\n"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(data_content)
        tmp_file_path = tmp_file.name

    try:
        # Action: Initialize the lookup table using TextFileIndex
        # Mirrors: torch.set_default_dtype(torch.float64) followed by z = x @ y
        # We explicitly use TextFileIndex.LINE_NUMBER (expects int64) and 
        # TextFileIndex.WHOLE_LINE (expects string) to test type handling.
        initializer = tf.lookup.TextFileInitializer(
            filename=tmp_file_path,
            key_dtype=tf.int64,
            key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            value_dtype=tf.string,
            value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            delimiter="\n"
        )

        table = tf.lookup.StaticHashTable(initializer, default_value="")

        # Verification: Access specific elements
        # Mirrors: print(z[0, 0])
        # We access index 0 and index 2 to ensure the mapping works as expected.
        key_0 = tf.constant(0, dtype=tf.int64)
        result_0 = table.lookup(key_0)
        
        key_2 = tf.constant(2, dtype=tf.int64)
        result_2 = table.lookup(key_2)

        # Assertions
        assert tf.equal(result_0, tf.constant("alpha", dtype=tf.string)).numpy(), \
            f"Expected 'alpha' at index 0, got {result_0.numpy()}"
        
        assert tf.equal(result_2, tf.constant("gamma", dtype=tf.string)).numpy(), \
            f"Expected 'gamma' at index 2, got {result_2.numpy()}"

        print("Test passed: TextFileIndex handled int64 keys and string values correctly.")

    finally:
        # Cleanup
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

if __name__ == "__main__":
    test_text_file_index_type_handling()