import torch

flag = True


def fn(x):
    x = x + 1
    # Replaced torch._dynamo.graph_break with torch.utils.checkpoint.set_checkpoint_debug_enabled
    torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
    x = x + 2
    if flag:
        # Replaced torch.no_grad context with torch.utils.checkpoint.set_checkpoint_debug_enabled
        torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
        torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
    else:
        torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
        torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
    return x + 4


# Replaced torch.ones with torch.linspace
fn(torch.linspace(1, 1, 3))
flag = False
fn(torch.linspace(1, 1, 3))