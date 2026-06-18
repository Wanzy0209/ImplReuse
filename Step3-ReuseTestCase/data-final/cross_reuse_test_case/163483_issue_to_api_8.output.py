import tensorflow as tf
import os
import tempfile

def test_textfileindex_ordering():
    """
    Test case for tf.lookup.TextFileIndex based on the logic of preserving 
    data ordering/layout, analogous to the PyTorch all_gather memory format bug.
    
    The original bug checks if the memory format (channels_last) is preserved 
    after an all_gather operation. This test checks if the data layout (specific 
    indices or whole lines) is preserved when using TextFileIndex to initialize 
    a lookup table.
    """
    
    # Create a temporary file with structured data
    # Format: ID,Name,Value
    data_content = """1,apple,10
2,banana,20
3,cherry,30
4,date,40
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(data_content)
        filename = f.name

    try:
        # --- Test Case 1: Specific Column Indexing ---
        # Analogous to checking if specific tensor strides (channels_last) are preserved.
        # We expect the key to be column 0 (ID) and value to be column 2 (Value).
        
        initializer_idx = tf.lookup.TextFileInitializer(
            filename=filename,
            key_dtype=tf.int64,
            key_index=tf.lookup.TextFileIndex(0),  # First column
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex(2), # Third column
            delimiter=","
        )
        
        table_idx = tf.lookup.StaticHashTable(initializer_idx, default_value=-1)
        
        # Verify that the mapping respects the defined indices
        key_1 = tf.constant(1, tf.int64)
        val_1 = table_idx.lookup(key_1)
        assert val_1.numpy() == 10, f"Expected 10, got {val_1.numpy()}"
        
        key_4 = tf.constant(4, tf.int64)
        val_4 = table_idx.lookup(key_4)
        assert val_4.numpy() == 40, f"Expected 40, got {val_4.numpy()}"
        
        print("Test Case 1 Passed: Specific column indices preserved correctly.")

        # --- Test Case 2: Whole Line Indexing ---
        # Analogous to checking raw storage preservation in the original bug.
        # We expect the value to be the entire raw line content.
        
        initializer_whole = tf.lookup.TextFileInitializer(
            filename=filename,
            key_dtype=tf.int64,
            key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            value_dtype=tf.string,
            value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            delimiter=","
        )
        
        table_whole = tf.lookup.StaticHashTable(initializer_whole, default_value="")
        
        # Verify that the raw line content is preserved exactly
        line_0 = table_whole.lookup(tf.constant(0, tf.int64))
        assert line_0.numpy() == b"1,apple,10", f"Expected b'1,apple,10', got {line_0.numpy()}"
        
        line_2 = table_whole.lookup(tf.constant(2, tf.int64))
        assert line_2.numpy() == b"3,cherry,30", f"Expected b'3,cherry,30', got {line_2.numpy()}"
        
        print("Test Case 2 Passed: Whole line content preserved correctly.")

    finally:
        # Clean up the temporary file
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_textfileindex_ordering()