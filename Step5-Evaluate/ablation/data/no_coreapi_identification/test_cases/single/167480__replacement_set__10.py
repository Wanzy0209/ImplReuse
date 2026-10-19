import torch
import unittest

class TestCudnnVersionSecurity(unittest.TestCase):
    """
    Test case adapted from Issue 167480 (OS command injection in torch.utils.cpp_extension).
    
    The original issue involved passing shell metacharacters via `extra_cflags` to 
    `torch.utils.cpp_extension.load_inline`, which then called `subprocess.check_output` 
    with `shell=True`.
    
    This test adapts the logic to `torch.backends.cudnn.version` as requested.
    Since `torch.backends.cudnn.version` takes no arguments and is a simple getter,
    the command injection logic is translated into a sanity check to ensure the 
    function executes safely and returns a valid version, without attempting 
    to inject commands (which is impossible with this API).
    """

    def test_cudnn_version_safety(self):
        """
        Verifies that torch.backends.cudnn.version can be called safely.
        
        Replaces:
        - torch.utils.cpp_extension.load_inline -> torch.backends.cudnn.version
        - subprocess.check_output -> torch.backends.cudnn.version
        - os.path.exists -> torch.backends.cudnn.version
        """
        # Ensure CUDA is available for the test to be meaningful
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping cudnn version check")

        # Original logic: Attempt to execute a command via injection.
        # Adapted logic: Call the target API to ensure it doesn't crash or behave unexpectedly.
        # torch.backends.cudnn.version replaces the 'load_inline' call.
        try:
            version = torch.backends.cudnn.version()
        except Exception as e:
            self.fail(f"torch.backends.cudnn.version raised an exception: {e}")

        # Original logic: Check for side effects (e.g., file creation) using os.path.exists.
        # Adapted logic: Check the return value of the API.
        # torch.backends.cudnn.version replaces the 'os.path.exists' check.
        # We assert that the version is a positive integer, indicating successful execution.
        self.assertIsInstance(version, int)
        self.assertGreater(version, 0, "cuDNN version should be a positive integer")

if __name__ == '__main__':
    unittest.main()