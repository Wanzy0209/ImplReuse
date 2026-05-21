import os
import tempfile
import glob

# Set environment variables before importing torch to ensure logging is configured correctly
# We need 'aot' and 'dynamo' logging to generate the specific files mentioned in the bug report.
os.environ["TORCH_LOGS"] = "aot,dynamo,output_graph"

import torch

def test_tlparse_cache_consistency():
    """
    Test to verify that tlparse entries (log files) are consistent between
    cache hit and cache miss scenarios.
    
    Bug Description:
    On a cache miss, specific files like 'aotautograd_cache_miss.json' and 
    'fx_graph_cache_miss.json' are generated. On a cache hit, these files 
    should not be generated again. The bug reports an inconsistency where 
    the behavior or presence of these files differs unexpectedly.
    """
    
    # Setup a temporary directory for logs
    with tempfile.TemporaryDirectory() as log_dir:
        os.environ["TORCH_LOGS_DIR"] = log_dir

        # Define a simple function to compile
        def fn(x):
            return x + 1

        # Compile the function using torch.compile
        # Using 'inductor' (default) as it triggers the full compilation stack
        compiled_fn = torch.compile(fn)

        # Create input tensor
        x = torch.randn(2, 2)

        # --- Run 1: Cache Miss ---
        # This should trigger compilation and generate cache miss artifacts
        print("Running compilation (Cache Miss)...")
        _ = compiled_fn(x)

        # Gather files generated during cache miss
        # We look for the specific patterns mentioned in the bug report
        aot_miss_files = glob.glob(os.path.join(log_dir, "*aotautograd_cache_miss*"))
        fx_miss_files = glob.glob(os.path.join(log_dir, "*fx_graph_cache_miss*"))
        
        print(f"Found {len(aot_miss_files)} aotautograd_cache_miss files after miss.")
        print(f"Found {len(fx_miss_files)} fx_graph_cache_miss files after miss.")

        # --- Run 2: Cache Hit ---
        # This should reuse the compiled graph and NOT generate new cache miss artifacts
        print("Running inference (Cache Hit)...")
        _ = compiled_fn(x)

        # Gather files generated after the second run
        aot_miss_files_after_hit = glob.glob(os.path.join(log_dir, "*aotautograd_cache_miss*"))
        fx_miss_files_after_hit = glob.glob(os.path.join(log_dir, "*fx_graph_cache_miss*"))

        print(f"Found {len(aot_miss_files_after_hit)} aotautograd_cache_miss files after hit.")
        print(f"Found {len(fx_miss_files_after_hit)} fx_graph_cache_miss files after hit.")

        # --- Assertions ---
        # 1. Verify that cache miss files were indeed created during the first run
        assert len(aot_miss_files) > 0, "Expected aotautograd_cache_miss files on cache miss, but found none."
        assert len(fx_miss_files) > 0, "Expected fx_graph_cache_miss files on cache miss, but found none."

        # 2. Verify that no NEW cache miss files were created during the cache hit
        # This checks for the inconsistency where 'miss' files might appear on a 'hit'
        assert len(aot_miss_files) == len(aot_miss_files_after_hit), \
            "Inconsistency detected: New aotautograd_cache_miss files were generated on cache hit."
        assert len(fx_miss_files) == len(fx_miss_files_after_hit), \
            "Inconsistency detected: New fx_graph_cache_miss files were generated on cache hit."

        print("Test passed: tlparse entries are consistent between cache hit and miss.")

if __name__ == "__main__":
    test_tlparse_cache_consistency()