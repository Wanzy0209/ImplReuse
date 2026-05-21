import torch
import tensorflow as tf
import os
import tempfile
import shutil

# Setup a temporary directory for file operations
tmp_dir = tempfile.mkdtemp()

def fuzzed_program(file_path, content):
    # Adapted logic: Instead of torch.unique and matmul, we use the target API tf.io.write_file.
    # The core logic being tested is the consistency of the operation execution.
    # In the original bug, eager and compiled modes diverged. Here we verify
    # that tf.io.write_file behaves consistently in both modes.
    
    # Ensure inputs are tensors if they aren't already
    path_tensor = tf.convert_to_tensor(file_path)
    content_tensor = tf.convert_to_tensor(content)
    
    # Perform the write operation
    # Note: tf.io.write_file is stateful (I/O), so we check side effects (file existence/content)
    op_result = tf.io.write_file(path_tensor, content_tensor)
    return op_result

# Setup arguments
file_name = "test_output.txt"
full_path = os.path.join(tmp_dir, file_name)
content_data = "Test content for divergence check"

args = (full_path, content_data)

# 1. Test Eager Mode
print("Testing Eager Mode...")
try:
    # Clean up before run
    if os.path.exists(full_path):
        os.remove(full_path)
        
    result_eager = fuzzed_program(*args)
    
    # Verify side effect
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            read_content = f.read()
        if read_content == content_data:
            print(' eager success')
        else:
            print(f' eager content mismatch: {read_content}')
    else:
        print(' eager failed to write file')
except Exception as e:
    print(f' eager error: {e}')

# 2. Test Compiled Mode (tf.function)
print("\nTesting Compiled Mode (tf.function)...")
try:
    # Clean up before run
    if os.path.exists(full_path):
        os.remove(full_path)

    # Compile the program
    compiled_program = tf.function(fuzzed_program)
    
    result_compiled = compiled_program(*args)
    
    # Verify side effect
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            read_content = f.read()
        if read_content == content_data:
            print(' compile success')
        else:
            print(f' compile content mismatch: {read_content}')
    else:
        print(' compile failed to write file')
except Exception as e:
    print(f' compile error: {e}')

# Cleanup
shutil.rmtree(tmp_dir)