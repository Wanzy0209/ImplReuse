import torch
import os
import tempfile

# Handle environment/dependency issues (e.g., missing GLIBCXX)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: {e}")
    print("TensorFlow could not be imported. This is likely due to a missing system dependency (GLIBCXX).")
    exit(0)

# Create a temporary directory for the test
temp_dir = tempfile.mkdtemp()
file_path = os.path.join(temp_dir, "test_output.txt")

# Define the eager version
def eager_fn(path, content):
    return tf.io.write_file(path, content)

# Define the compiled version (tf.function is analogous to torch.compile)
@tf.function
def compiled_fn(path, content):
    return tf.io.write_file(path, content)

# Prepare inputs
# Mimicking the "empty" tensor aspect from the original bug
# Original: y = torch.empty((), dtype=torch.complex64)
# Adapted: An empty string tensor
content = tf.constant("", dtype=tf.string)

# Test Eager
try:
    eager_fn(file_path, content)
    print("eager success")
except Exception as e:
    print(f"eager failed: {e}")

# Test Compiled
try:
    compiled_fn(file_path, content)
    print("compiler success")
except Exception as e:
    print(f"compiler failed: {e}")

# Cleanup
if os.path.exists(file_path):
    os.remove(file_path)
os.rmdir(temp_dir)