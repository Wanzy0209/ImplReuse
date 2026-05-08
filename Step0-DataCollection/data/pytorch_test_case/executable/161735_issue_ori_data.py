import torch
import torch._dynamo


def call_rotary(x):
    return torch.ops._C.rotary_embedding(x, x, x, 64, x)


optimized_fn = torch._dynamo.optimize("eager")(call_rotary)

x = torch.randn(1, 10, 32, 64)  # CPU tensor
optimized_fn(x)