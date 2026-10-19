import torch
import torch.distributed

# Define a dummy op to make the impl_abstract decorator valid
try:
    torch.library.define("custom::dummy_op", "(Tensor x) -> Tensor")
except Exception:
    pass

flag = True

# Replacement: torch.compile -> torch.library.impl_abstract
@torch.library.impl_abstract("custom::dummy_op", "default")
def fn(x):
    x = x + 1
    # Replacement: torch._dynamo.graph_break -> torch.distributed.is_torchelastic_launched
    torch.distributed.is_torchelastic_launched()
    x = x + 2
    if flag:
        # Replacement: torch.no_grad -> torch.distributed.is_torchelastic_launched
        # Note: torch.no_grad is a context manager, is_torchelastic_launched is a function.
        # Replacing the API call directly.
        torch.distributed.is_torchelastic_launched()
        torch.distributed.is_torchelastic_launched()
    else:
        torch.distributed.is_torchelastic_launched()
        torch.distributed.is_torchelastic_launched()
    return x + 4

# Replacement: torch.ones -> torch.normal
# torch.ones(3) creates a tensor of 1s. torch.normal creates a tensor of random numbers.
# We use mean=0, std=1, size=(3,) to match the shape.
input_tensor = torch.normal(0, 1, (3,))

fn(input_tensor)
flag = False
fn(input_tensor)