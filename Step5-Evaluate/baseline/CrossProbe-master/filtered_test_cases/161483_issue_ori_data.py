import torch
if torch.xpu.is_available():
    # Reproduce test_default_priority_order_xpu failure
    pass