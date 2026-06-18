import torch

@torch.compile(fullgraph=True)
def fn(mask, hidden):
    # Create a new tensor inside the graph, similar to the original bug report
    hidden_states = hidden.new_zeros([1, 512, 3072])
    
    # Use torch.any to get a scalar boolean
    # This replaces the sum().item() in the original bug to test data-dependent slicing
    has_data = torch.any(mask).item()
    
    # Convert boolean to integer for slicing (0 or 1)
    # This creates a data-dependent slice that triggers the inductor path
    slice_len = int(has_data)
    
    return hidden_states[:, :slice_len]

torch._dynamo.config.capture_scalar_outputs = True

# Setup inputs
mask = torch.tensor([True, False]).cuda()
hidden = torch.randn((1, 512, 4096)).cuda()

# Run the function
result = fn(mask, hidden)

# Basic assertion to verify execution
assert result.shape == (1, 1, 3072)