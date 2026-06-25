import torch
from torch.nn.attention.flex_attention import flex_attention
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
flex_attention(query, key, value)