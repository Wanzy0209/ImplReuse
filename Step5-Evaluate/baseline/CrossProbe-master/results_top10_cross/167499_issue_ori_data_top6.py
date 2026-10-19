```python
import os
import shutil
import subprocess

# Conversion Note: TensorFlow does not provide runtime utilities for C++ compiler detection
# as it relies on Bazel for building custom ops. We use standard Python to replicate the behavior.

def get_cxx_compiler():
    # PyTorch checks CXX environment variable first
    cxx = os.environ.get('CXX')
    if cxx:
        return cxx
    # Fallback to system default
    return shutil.which('c++') or shutil.which('g++') or 'c++'

def check_compiler_is_gcc(compiler_path):
    try:
        # Check compiler version string for 'gcc'
        result = subprocess.run([compiler_path, '--version'], capture_output=True, text=True)
        return 'gcc' in result.stdout.lower()
    except Exception:
        return False

compiler = get_cxx_compiler()  # Returns /usr/bin/g++-13
result = check_compiler_is_gcc(compiler)

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")  # False (incorrect!)
```