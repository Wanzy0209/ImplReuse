# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.nn.attention.flex_attention import flex_attention

inductor = torch.compile(flex_attention, fullgraph=True, backend="inductor")

with torch.device("cuda"):
    q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

y = inductor(q, k, v, enable_gqa=True)
y.backward(torch.randn_like(y))