import torch
import math

Q = torch.randn(2, 3, 4)
K = torch.randn(2, 3, 4)
attn_mask = torch.zeros(2, 3, 3)

# Current broken example (syntax error)
# attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * attn_mask, dim=-1)

# Corrected version
scale_factor = 1.0 / math.sqrt(Q.size(-1))
attn_weight = torch.softmax(
    (Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1
)