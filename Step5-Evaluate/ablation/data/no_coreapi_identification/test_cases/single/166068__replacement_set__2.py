import torch

# Original logic: version.parse(torch.version.hip)
# Replacement: torch.backends.mkldnn.is_available()
# Note: is_available takes no arguments, so the argument torch.version.hip is dropped.
print(torch.backends.mkldnn.is_available())

# Original logic (from bug description): version.parse(torch.version.cuda)
# Replacement: torch.backends.mkldnn.is_available()
# Note: torch.version.cuda is replaced by is_available, and version.parse is replaced by is_available.
# Semantics translation: is_available takes no arguments, so we just call it.
print(torch.backends.mkldnn.is_available())