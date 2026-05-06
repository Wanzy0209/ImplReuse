import torch

from torch._dynamo.functional_export import _dynamo_graph_capture_for_export

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class FlexAttentionModule(torch.nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query, key, value, block_mask=None):
        return flex_attention(query, key, value, block_mask=block_mask)

flex_model = FlexAttentionModule(head_dim=64)

batch_size = 2
num_heads = 4
seq_len = 128
head_dim = 64

query = torch.randn(batch_size, num_heads, seq_len, head_dim)
key = torch.randn(batch_size, num_heads, seq_len, head_dim)
value = torch.randn(batch_size, num_heads, seq_len, head_dim)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(
    causal_mask, batch_size, num_heads, seq_len, seq_len, device="cpu"
)

flex_inputs = (query, key, value)
flex_kwargs = {"block_mask": block_mask}

eager_out = flex_model(*flex_inputs, **flex_kwargs)

with torch._dynamo.config.patch(install_free_tensors=True):
    gm = _dynamo_graph_capture_for_export(flex_model)(*flex_inputs, **flex_kwargs)