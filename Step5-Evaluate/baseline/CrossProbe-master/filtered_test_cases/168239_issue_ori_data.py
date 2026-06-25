import torch
from torch._inductor import config as inductor_config

# Current behavior - either fully disabled or enabled
# inductor_config.force_disable_caches = True  # Disables cache completely
# inductor_config.force_disable_caches = False # Loads from disk

# Need new option like:
# inductor_config.ignore_disk_cache = True