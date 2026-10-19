import torch

# Original test case logic: version.parse(torch.version.hip)
# Replacements applied:
# - packaging.version.parse -> torch.distributed.is_available
# - torch.version.cuda -> torch.distributed.is_available (not present in original snippet, but logic adapted)
# Semantics translated: Parse version string -> Check availability status
# Note: torch.distributed.is_available() takes no arguments, so torch.version.hip is removed.

is_available = torch.distributed.is_available()
print(is_available)

# Basic assertion to ensure the API call is successful and returns a boolean
assert isinstance(is_available, bool), "torch.distributed.is_available should return a boolean"