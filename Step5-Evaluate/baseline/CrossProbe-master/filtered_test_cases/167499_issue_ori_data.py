from torch.utils.cpp_extension import get_cxx_compiler, check_compiler_is_gcc
compiler = get_cxx_compiler()
result = check_compiler_is_gcc(compiler)
print(f'Compiler: {compiler}')
print(f'Detected as GCC: {result}')