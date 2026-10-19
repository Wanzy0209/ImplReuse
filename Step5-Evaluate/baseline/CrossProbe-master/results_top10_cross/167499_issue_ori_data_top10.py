```python
import tensorflow as tf

# Conversion: PyTorch's get_cxx_compiler is not directly available in TensorFlow.
# We use tf.sysconfig.get_build_info() to approximate the compiler identity.
build_info = tf.sysconfig.get_build_info()
# Note: The key 'cpu_compiler' may vary or be absent in some builds.
compiler = build_info.get('cpu_compiler', 'g++')  # Returns compiler name (e.g., 'gcc-9')

# Conversion: PyTorch's check_compiler_is_gcc checks the compiler path.
# Here we check if the compiler string indicates GCC.
result = 'gcc' in compiler.lower()

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")
```