attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)  # Already query.dtype
if is_causal:
    # ...
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)  # Redundant! Already query.dtype, and result not assigned