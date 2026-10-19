```python
import tensorflow as tf

# Conversion Note: TensorFlow does not have a direct equivalent to torch.utils.cpp_extension
# for retrieving the C++ compiler path. TensorFlow uses Bazel for building custom ops.
# We use tf.sysconfig to get the compiler information used to build TensorFlow itself.

# Get the compiler information (e.g., "GCC 9.3.0")
compile_info = tf.sysconfig.get_compile_info()
compiler = compile_info.get('cpu_compiler', 'Unknown')

# Check if the compiler string indicates GCC
result = 'gcc' in compiler.lower()

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")  # False (incorrect!)
```