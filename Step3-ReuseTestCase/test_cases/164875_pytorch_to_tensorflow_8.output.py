import torch
import tensorflow as tf
import os
import tempfile

# Setup
tf.random.set_seed(1014698)

def fuzzed_program(filename, content):
    # Original logic: torch.add(var_node_1, var_node_2)
    # Adapted logic: tf.io.write_file(filename, content)
    # We verify that the write operation executes correctly in both modes.
    tf.io.write_file(filename, content)

# Prepare inputs
# We use a temporary directory to handle file I/O safely
tmp_dir = tempfile.mkdtemp()
file_path = os.path.join(tmp_dir, "test_tf_write_file.txt")
content_tensor = tf.constant("Test content for tf.io.write_file")

# Run Eager
print('Running eager execution...')
try:
    fuzzed_program(file_path, content_tensor)
    if tf.io.gfile.exists(file_path):
        print(' eager success')
    else:
        print(' eager failed: file not created')
except Exception as e:
    print(f' eager failed: {e}')

# Run Compiled (tf.function)
# Note: fullgraph=True in torch.compile is roughly equivalent to jit_compile=True in tf.function
compiled_program = tf.function(fuzzed_program, jit_compile=True)

# Clean up before compiled run to ensure it actually writes
if tf.io.gfile.exists(file_path):
    tf.io.gfile.remove(file_path)

print('Running compiled execution...')
try:
    compiled_program(file_path, content_tensor)
    if tf.io.gfile.exists(file_path):
        print(' compile success')
    else:
        print(' compile failed: file not created')
except Exception as e:
    print(f' compile failed: {e}')

# Cleanup
if tf.io.gfile.exists(file_path):
    tf.io.gfile.remove(file_path)
os.rmdir(tmp_dir)