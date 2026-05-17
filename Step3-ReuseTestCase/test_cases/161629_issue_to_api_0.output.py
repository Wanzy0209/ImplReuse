import torch
import torch.utils.cpp_extension as cpp_ext
import warnings
import os
from unittest.mock import patch

def test_compiler_abi_warning_distributed_awareness():
    """
    Test that warnings in get_compiler_abi_compatibility_and_version
    respect distributed training settings (only print on rank 0).
    
    This test mirrors the logic required by Issue 161629, where warnings
    in cpp_extension should be rank-aware to avoid log spam in multi-GPU setups.
    """
    # Ensure the environment variable that suppresses the check is not set,
    # so we can trigger the warning logic.
    os.environ.pop('TORCH_DONT_CHECK_COMPILER_ABI', None)

    # Mock the platform check to return False, forcing the warning path
    # in get_compiler_abi_compatibility_and_version.
    with patch('torch.utils.cpp_extension.check_compiler_ok_for_platform', return_value=False):
        
        # Scenario 1: Rank 0 (Main process)
        # Expected: Warning should be emitted.
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_rank', return_value=0):
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Call the similar API
                is_compatible, version = cpp_ext.get_compiler_abi_compatibility_and_version('g++')
                
                # Verify warning was raised
                assert len(w) == 1
                assert "compiler" in str(w[0].message).lower()

        # Scenario 2: Rank 7 (Worker process)
        # Expected: Warning should be suppressed (based on the fix requested in the issue).
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_rank', return_value=7):
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Call the similar API
                is_compatible, version = cpp_ext.get_compiler_abi_compatibility_and_version('g++')
                
                # Verify warning was NOT raised
                assert len(w) == 0, f"Warning should be suppressed on non-zero ranks, but got: {[str(x.message) for x in w]}"

if __name__ == "__main__":
    test_compiler_abi_warning_distributed_awareness()
    print("Test passed.")