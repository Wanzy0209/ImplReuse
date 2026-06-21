import tensorflow as tf
import sys

def get_compile_flags_wrapper():
    """
    Wrapper function to mimic the function call pattern in the original issue.
    This retrieves the compilation flags required to build custom TensorFlow ops.
    """
    return tf.sysconfig.get_compile_flags()

def main():
    # The original test runs a loop to check behavior (GIL release).
    # Here we run a loop to ensure the API returns consistent and correct results.
    # Note: Unlike the CUDA kernels in the PyTorch issue, this is a CPU-bound
    # Python utility, so holding the GIL is expected behavior.
    
    print("Testing tf.sysconfig.get_compile_flags consistency...")
    
    for i in range(10):
        flags = get_compile_flags_wrapper()

        # Verification 1: Ensure the return type is a list
        assert isinstance(flags, list), f"Expected list, got {type(flags)}"

        # Verification 2: Ensure the list is not empty
        assert len(flags) > 0, "Compilation flags list should not be empty"

        # Verification 3: Check for specific flags based on the API implementation
        # The implementation appends include paths starting with '-I'
        has_include = any(flag.startswith('-I') for flag in flags)
        assert has_include, "Flags should include include directory paths (-I...)"

        # The implementation appends the CXX ABI flag
        has_abi = any('_GLIBCXX_USE_CXX11_ABI' in flag for flag in flags)
        assert has_abi, "Flags should define the CXX11 ABI macro"

        # The implementation appends a C++ standard flag (e.g., -std=c++14)
        # Fix: Relaxed the check to look for 'std=c++' substring to handle variations
        # like '-std=c++' (single dash) or '--std=c++' (double dash).
        has_std = any('std=c++' in flag for flag in flags)
        assert has_std, "Flags should specify the C++ standard version"

        print(f"Iteration {i+1}: Passed. Found {len(flags)} flags.")

    print("All tests passed.")

if __name__ == "__main__":
    main()