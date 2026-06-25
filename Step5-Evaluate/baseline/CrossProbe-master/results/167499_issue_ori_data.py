```python
import os
import shutil
import subprocess

# Conversion: TensorFlow does not have a direct runtime equivalent for PyTorch's cpp_extension utilities.
# TensorFlow custom ops are typically built using Bazel or setup.py, not runtime compilation.
# We implement the logic using standard Python libraries to mimic the behavior.

def get_cxx_compiler():
    # Mimic PyTorch behavior: Check CXX env var, then fallback to 'g++' or 'c++'
    cxx = os.environ.get('CXX')
    if cxx:
        return cxx
    return shutil.which('g++') or shutil.which('c++')

def check_compiler_is_gcc(compiler):
    if not compiler:
        return False
    try:
        # Check compiler version output for 'gcc' string
        result = subprocess.run([compiler, '--version'], capture_output=True, text=True)
        return 'gcc' in result.stdout.lower()
    except (FileNotFoundError, OSError):
        return False

compiler = get_cxx_compiler()  # Returns path to g++ or c++
result = check_compiler_is_gcc(compiler)

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")
```