```python
import os
import shutil
import subprocess

# Conversion: TensorFlow does not expose direct compiler utilities like PyTorch.
# We use environment variables and standard Python libraries to emulate this behavior.
def get_cxx_compiler():
    # Priority: CXX env var -> CC env var -> system default g++
    return os.environ.get('CXX') or os.environ.get('CC') or shutil.which('g++')

def check_compiler_is_gcc(compiler_path):
    if not compiler_path:
        return False
    try:
        # Run compiler --version to identify it
        output = subprocess.check_output([compiler_path, '--version'], stderr=subprocess.STDOUT, text=True)
        return 'gcc' in output.lower() or 'g++' in output.lower()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False

compiler = get_cxx_compiler()
result = check_compiler_is_gcc(compiler)

print(f"Compiler: {compiler}")
print(f"Detected as GCC: {result}")
```