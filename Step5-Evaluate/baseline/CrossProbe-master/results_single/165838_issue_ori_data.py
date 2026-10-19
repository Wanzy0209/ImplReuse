# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from typing import Union
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

compiled_flex_attention_hf = torch.compile(flex_attention,fullgraph=True)
def block_mask_fn_1(batch: Tensor, h: Tensor, q: Tensor, kv: Tensor):
    causal_mask = (q >= kv)
    return causal_mask

def block_mask_fn_2(batch: Tensor, h: Tensor, q: Tensor, kv: Tensor):
    causal_mask = hand_made_causal_mask[q, kv]
    return causal_mask

seq_len = 2048
query_head = 16
kv_head = 8
head_dim = 128

kernel_options: dict[str, Union[int, bool]] = {
        "FORCE_USE_FLEX_ATTENTION": True,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "IS_DIVISIBLE": False,
    }

attention_mask = torch.ones((1, seq_len), dtype=torch.bool).cuda()
# hand made causal mask
hand_made_causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device))

for i in range(500):
    query_hf = torch.randn((1, query_head, seq_len, head_dim),dtype=torch.bfloat16).cuda()
    key_hf = torch.randn((1, kv_head, seq_len, head_dim),dtype=torch.bfloat16).cuda()
    value_hf = torch.randn((1, kv_head, seq_len, head_dim),dtype=torch.bfloat16).cuda()

    print('-'*50)
    # first method to create block mask
    block_mask_hf_1 = create_block_mask(block_mask_fn_1, 1, None, seq_len, seq_len, BLOCK_SIZE=(16, 16), device='cuda')
    y1 = compiled_flex_attention_hf(query_hf,key_hf,
            value_hf,
            block_mask=block_mask_hf_1,
            scale=None,
            enable_gqa=True,
            score_mod=None,
            kernel_options=kernel_options,
        )
    # second method to create block mask
    block_mask_hf_2 = create_block_mask(block_mask_fn_2, 1, None, seq_len, seq_len, BLOCK_SIZE=(16, 16), device='cuda')
    y2 = compiled_flex_attention_hf(
            query_hf,
            key_hf,
            value_hf,
            block_mask=block_mask_hf_2,
            scale=None,
            enable_gqa=True,
            score_mod=None,
            kernel_options=kernel_options,
        )
    print('Different element count',torch.sum((y1-y2)!=0), 'Sum difference', (y1.float().sum()-y2.float().sum()).abs())