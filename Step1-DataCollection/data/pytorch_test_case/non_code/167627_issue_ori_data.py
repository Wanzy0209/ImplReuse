attn_weight = torch.softmax(
    (Q @ K.transpose(-2, -1) * attn_mask, dim=-1
)