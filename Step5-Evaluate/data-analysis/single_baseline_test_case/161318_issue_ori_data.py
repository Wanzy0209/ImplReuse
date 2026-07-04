# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

@torch.compile(fullgraph=True)
def fn(encoder_attention_mask, encoder_hidden_states):
    encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
    text_len = encoder_attention_mask.sum().item()
    encoder_hidden_states = encoder_hidden_states[:, :text_len]

torch._dynamo.config.capture_scalar_outputs = True
mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
hidden = torch.randn((1, 512, 4096)).cuda()

fn(mask, hidden)