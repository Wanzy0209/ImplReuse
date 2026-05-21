import torch
import tensorflow as tf
import tempfile
import os
import shutil

def test_match_filenames_once_behavior():
    """
    Adapted test case for tf.io.match_filenames_once based on the torch.full bug report.
    
    Original Bug Logic:
    1. Call function with Input A -> Result A.
    2. Call function with Input B -> Result B (Expected), Result A (Buggy).
    
    Adaptation:
    1. Call tf.io.match_filenames_once with Pattern A -> List A.
    2. Call tf.io.match_filenames_once with Pattern B -> List B.
    3. Verify that List A != List B.
    """
    
    # Setup: Create a temporary directory with dummy files to match
    tmpdir = tempfile.mkdtemp()
    try:
        # Create files corresponding to the "values" in the PyTorch example (5.0 and 10.0)
        file_5 = os.path.join(tmpdir, "data_5.txt")
        file_10 = os.path.join(tmpdir, "data_10.txt")
        
        with open(file_5, 'w') as f:
            f.write("content")
        with open(file_10, 'w') as f:
            f.write("content")

        # tf.io.match_filenames_once relies on graph mode and collections
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        # Define the inputs (patterns)
        pattern1 = os.path.join(tmpdir, "data_5.txt")
        pattern2 = os.path.join(tmpdir, "data_10.txt")

        # Call the API with different inputs.
        # Note: We use different 'name' arguments because match_filenames_once 
        # creates a persistent variable. Using the same name would return the 
        # same variable reference, which is intended behavior in TF but would 
        # obscure the test for input sensitivity.
        files1 = tf.io.match_filenames_once(pattern1, name="match_files_1")
        files2 = tf.io.match_filenames_once(pattern2, name="match_files_2")

        with tf.compat.v1.Session() as sess:
            # Initialize local variables (where the matched filenames are stored)
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())

            # Get results
            result1 = sess.run(files1)
            result2 = sess.run(files2)

            print("Result 1:", result1)
            print("Result 2:", result2)

            # Verify that changing the input pattern changes the output.
            # In the PyTorch bug, result2 would be equal to result1.
            assert result1 != result2, "API returned cached result for different input pattern"
            assert file_5 in result1
            assert file_10 in result2

    finally:
        # Cleanup temporary directory
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    test_match_filenames_once_behavior()