import torch

def custom_op(input):
    return input.view_as(input)  # Returns a view, violates assumption

# Register custom op (hypothetical)
torch.library.define('custom::op', '(Tensor x) -> Tensor')
torch.library.impl('custom::op', 'cpu')(custom_op)