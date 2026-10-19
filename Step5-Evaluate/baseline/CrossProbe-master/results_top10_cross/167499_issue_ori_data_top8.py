```python
import shutil
import subprocess

# Conversion: TensorFlow does not have a direct runtime equivalent for torch.utils.cpp_extension.
# We use standard Python libraries to replicate the compiler detection logic.

def get_cxx_compiler():
    # Try to find the default C++ compiler (g++ or clang++)
    return shutil.which('g++') or shutil.which('clang++')

def check_compiler_is_gcc(compiler_path):
    if not compiler_path:
        return False
    try:
        # Check the compiler version string to determine if it is GCC
        output = subprocess.check_output([compiler_path, '--version'], text=True)
        return 'gcc' in output.lower()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False

compiler = get_cxx_compiler()  # Returns /usr/bin/g++-13
result = check_compiler_is_gcc(compiler)

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")  # False (incorrect!)
```