```python
import sysconfig
import os
import shutil

# Conversion: TensorFlow does not expose C++ compiler utilities via tf API.
# We use Python's sysconfig to emulate this behavior.
def get_cxx_compiler():
    # Try to get the compiler used to build Python (often used for TF extensions)
    compiler = sysconfig.get_config_var('CXX')
    if compiler and os.path.exists(compiler):
        return compiler
    
    # Fallback to finding g++ or c++ in PATH
    return shutil.which('g++') or shutil.which('c++')

def check_compiler_is_gcc(compiler):
    if not compiler:
        return False
    # Check if 'gcc' is in the compiler name
    return 'gcc' in os.path.basename(compiler).lower()

compiler = get_cxx_compiler()  # Returns path to C++ compiler
result = check_compiler_is_gcc(compiler)

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")  # False (incorrect!)
```