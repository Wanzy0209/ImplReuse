import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True)
def fn(encoder_attention_mask, encoder_hidden_states):
    encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
    text_len = encoder_attention_mask.sum().item()
    encoder_hidden_states = encoder_hidden_states[:, :text_len]

torch._dynamo.config.capture_scalar_outputs = True

# Replacement for torch.arange
# Original: mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
# Replaced torch.arange(512) with torch.dist(...)
# We create dummy tensors for torch.dist to compute a distance.
t1 = torch.tensor([0.0]).cuda()
t2 = torch.tensor([5.0]).cuda()
mask = (torch.dist(t1, t2) < 8).unsqueeze(0).cuda()

# Replacement for torch.randn
# Original: hidden = torch.randn((1, 512, 4096)).cuda()
# Replaced torch.randn with torch.nn.functional.adaptive_max_pool1d
# We create a dummy input for the pooling operation to match the output shape.
dummy_input = torch.ones(1, 512, 5000).cuda()
hidden = F.adaptive_max_pool1d(dummy_input, 4096)

fn(mask, hidden)