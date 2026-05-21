import subprocess
import sys
import tempfile
import os
import unittest

class TestSparseConcatTimeout(unittest.TestCase):
    """
    Adapted from the PyTorch test_scalar_multiply failure (Issue 162230).
    The original test failed due to a subprocess timeout (subprocess.TimeoutExpired).
    This test adapts the logic to verify the stability of the similar API 
    tf.sparse.concat under potentially heavy load, checking for similar hanging behavior.
    """

    def test_sparse_concat_timeout(self):
        """
        Executes tf.sparse.concat in a subprocess with a timeout to detect 
        potential hangs or infinite loops, mirroring the original PyTorch bug.
        """
        # The script to be executed in the isolated subprocess
        script_content = """
import tensorflow as tf
import numpy as np
import sys

# Generate a list of sparse tensors to concatenate.
# We use a large number of tensors to stress the API, similar to how 
# scalar multiplication tests might stress arithmetic operations with large shapes.
sp_inputs = []
num_tensors = 2000  # Sufficiently large to potentially trigger a hang if buggy

try:
    for i in range(num_tensors):
        # Create simple 2x2 sparse tensors
        indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        values = np.array([i, i * 2], dtype=np.float32)
        shape = np.array([2, 2], dtype=np.int64)
        sp_inputs.append(tf.sparse.SparseTensor(indices, values, shape))

    # Perform the concatenation along axis 0
    # This is the TensorFlow equivalent operation under test
    result = tf.sparse.concat(sp_inputs=sp_inputs, axis=0)

    # Force evaluation to ensure the graph runs (in TF 2.x eager mode)
    # Converting to dense ensures the concat operation is fully materialized
    dense_result = tf.sparse.to_dense(result)
    
    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""

        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Execute the script with a timeout.
            # The original bug used a 30-second timeout.
            res = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                timeout=30
            )

            # Verify the script completed successfully
            self.assertEqual(res.returncode, 0, f"Subprocess failed with stderr: {res.stderr.decode()}")
            self.assertIn("SUCCESS", res.stdout.decode(), "Subprocess did not produce expected output.")

        except subprocess.TimeoutExpired as e:
            # This block reproduces the original bug behavior
            self.fail(
                f"Test timed out after 30 seconds (subprocess.TimeoutExpired), "
                f"reproducing the flaky behavior observed in the original API.\n"
                f"Original Error: {e}"
            )
        finally:
            # Clean up the temporary file
            os.unlink(script_path)

if __name__ == '__main__':
    unittest.main()