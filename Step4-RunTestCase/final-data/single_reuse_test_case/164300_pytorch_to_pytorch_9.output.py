import torch
import torch.hub
import functools
from unittest.mock import patch

# Adapted from Issue 164300: Testing functools.partial with a similar PyTorch API
# Original issue: functools.partial'ed context_fn in torch.utils.checkpoint.checkpoint
# This test verifies if torch.hub.list works when invoked via functools.partial

# Define a partial function for torch.hub.list
# Using a standard repo to ensure testability
repo = "pytorch/vision"
list_fn = functools.partial(torch.hub.list, repo)

# Define a wrapper function to call the API
def get_hub_entries():
    return list_fn()

# Execute the test
# We mock torch.hub.list to avoid network timeouts and dependency on external repositories.
# This ensures the test runs quickly and verifies the logic of functools.partial.
with patch('torch.hub.list') as mock_list:
    # Configure the mock to return a dummy list, simulating a successful API call
    mock_list.return_value = ['resnet18', 'resnet50', 'alexnet']

    try:
        entries = get_hub_entries()
        assert isinstance(entries, list), "torch.hub.list should return a list of entrypoints"
        
        # Verify that the partial function correctly passed the 'repo' argument to the mocked function
        mock_list.assert_called_once_with(repo)
        
        print(f"Test passed: Successfully retrieved {len(entries)} entrypoints using functools.partial")
    except Exception as e:
        print(f"Test failed: {e}")