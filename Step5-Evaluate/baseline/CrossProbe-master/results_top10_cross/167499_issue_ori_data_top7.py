```python
import tensorflow as tf
import shutil
import subprocess

# Conversion Note: TensorFlow does not have a direct runtime equivalent for 
# torch.utils.cpp_extension because it relies on Bazel for building C++ ops 
# rather than JIT compilation. We emulate the behavior using standard Python libraries.

def get_cxx_compiler():
    # Emulate torch.utils.cpp_extension.get_cxx_compiler
    return shutil.which('g++') or shutil.which('c++')

def check_compiler_is_gcc(compiler_path):
    # Emulate torch.utils.cpp_extension.check_compiler_is_gcc
    if not compiler_path:
        return False
    try:
        result = subprocess.run([compiler_path, '--version'], capture_output=True, text=True)
        return 'gcc' in result.stdout.lower()
    except Exception:
        return False

compiler = get_cxx_compiler()  # Returns path to g++ or c++
result = check_compiler_is_gcc(compiler)

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")
```