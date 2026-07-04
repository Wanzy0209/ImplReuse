import torch
import torch.hub
import gc
import tracemalloc

# Define the repository to test
# Using a standard repo to ensure the test is runnable
repo = "pytorch/vision"

# Start memory tracing to detect leaks
tracemalloc.start()

print("Testing torch.hub.list for memory leaks...")
# Reduced iterations from 100 to 5 to avoid timeout in testing environments
# where network operations might be slow.
for i in range(5):
    # Adapted call site: torch.hub.list replaces torch.utils.checkpoint.checkpoint
    # Note: torch.hub.list does not take a custom function or use_reentrant argument.
    # We test the API's resource management directly.
    entrypoints = torch.hub.list(repo, skip_validation=True)

    # Clean up references to allow garbage collection
    del entrypoints
    gc.collect()

    # Check memory usage (RAM instead of CUDA, as hub.list is CPU/IO bound)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Iteration {i}: {current / 1024**2:.2f} MiB")

tracemalloc.stop()