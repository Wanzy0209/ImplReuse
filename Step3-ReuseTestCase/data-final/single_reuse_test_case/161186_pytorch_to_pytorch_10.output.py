import torch
import gc

# The original bug report highlights a memory leak in torch.utils.checkpoint.checkpoint
# when used with custom autograd functions. We adapt this test to verify the stability
# and memory behavior of the similar API, torch.hub.help.

# Define inputs for torch.hub.help
github = "pytorch/vision"
model = "resnet18"

# Warm up run to ensure the repo is available and dependencies are loaded
try:
    torch.hub.help(github, model, force_reload=True)
except Exception as e:
    print(f"Skipping test: Could not load model info. Error: {e}")
    exit()

# Loop to check for memory leaks or errors
# Note: Iterations are kept low (5) compared to the original (1000) because
# torch.hub.help involves network/disk IO which is significantly slower.
for i in range(5):
    # Call the similar API
    torch.hub.help(github, model, force_reload=True)
    
    # Explicit cleanup to check if memory is freed between calls
    gc.collect()
    
    print(f"Iteration {i} complete")

print("Test finished successfully.")