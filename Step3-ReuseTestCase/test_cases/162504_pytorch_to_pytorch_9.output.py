import torch

# Define inputs for torch.hub.list
# Using a standard repo for testing purposes
repo = "pytorch/vision"

# Eager execution
# Replacing torch.utils.checkpoint.checkpoint with torch.hub.list
eager_out = torch.hub.list(repo, force_reload=False)

# Graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    # Adapted call site: torch.hub.list instead of torch.utils.checkpoint.checkpoint
    # Note: torch.hub.list returns a list of strings, not a tensor.
    # Gradient logic is removed as it does not apply to string lists.
    graph_out = torch.hub.list(repo, force_reload=False)

# Replay
g.replay()

# Assertion
# Since torch.hub.list is not a CUDA operation, it executes during the capture phase.
# We verify that the outputs match.
assert eager_out == graph_out, "Mismatch in outputs"
print("Eager output:", eager_out)
print("Graph output:", graph_out)