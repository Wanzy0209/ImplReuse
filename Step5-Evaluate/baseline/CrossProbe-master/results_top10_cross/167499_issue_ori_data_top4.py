```python
import tensorflow as tf

# Conversion: PyTorch's get_cxx_compiler returns the binary path.
# TensorFlow provides build configuration via tf.sysconfig.
build_info = tf.sysconfig.get_build_info()
compiler = build_info.get('cpu_compiler', 'unknown')

# Conversion: PyTorch checks the binary path. TensorFlow provides the compiler name
# directly in build info. We check if the string indicates GCC.
result = 'gcc' in compiler.lower()

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")  # False (incorrect!)
```