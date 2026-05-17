import os
import unittest
import warnings
from unittest import mock

import torch
import torch.utils.cpp_extension as cpp_ext


class TestCppExtensionDistributedWarning(unittest.TestCase):
    def test_include_paths_no_warning_on_non_zero_rank(self):
        """
        Test that calling include_paths (or related cpp_extension utilities)
        does not trigger the TORCH_CUDA_ARCH_LIST warning on non-zero ranks.
        
        This addresses the bug report where a warning about TORCH_CUDA_ARCH_LIST
        was printed on all ranks in a distributed setup.
        """
        # Save original state
        original_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
        
        try:
            # Ensure the environment variable is not set to trigger the warning condition
            if 'TORCH_CUDA_ARCH_LIST' in os.environ:
                del os.environ['TORCH_CUDA_ARCH_LIST']

            # Mock distributed environment to simulate rank 7 (as in the bug report)
            with mock.patch('torch.distributed.is_initialized', return_value=True), \
                 mock.patch('torch.distributed.get_rank', return_value=7):

                # Capture warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Call the similar API: include_paths
                    # We use device_type="cuda" to ensure CUDA-related paths are checked
                    paths = cpp_ext.include_paths(device_type="cuda")

                    # Verify that the specific warning about TORCH_CUDA_ARCH_LIST is NOT present
                    # The bug report indicates this warning should be rank-aware (only on rank 0)
                    # or suppressed unless in DEBUG/INFO mode.
                    warning_messages = [str(warning.message) for warning in w]
                    
                    for msg in warning_messages:
                        self.assertNotIn(
                            "TORCH_CUDA_ARCH_LIST", 
                            msg,
                            f"Warning about TORCH_CUDA_ARCH_LIST should not appear on rank 7. Found: {msg}"
                        )

        finally:
            # Restore environment variable
            if original_arch_list is not None:
                os.environ['TORCH_CUDA_ARCH_LIST'] = original_arch_list
            elif 'TORCH_CUDA_ARCH_LIST' in os.environ:
                del os.environ['TORCH_CUDA_ARCH_LIST']


if __name__ == '__main__':
    unittest.main()