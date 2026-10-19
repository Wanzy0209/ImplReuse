import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True)
def fn(encoder_attention_mask, encoder_hidden_states):
    encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
    text_len = encoder_attention_mask.sum().item()
    encoder_hidden_states = encoder_hidden_states[:, :text_len]
    return encoder_hidden_states

torch._dynamo.config.capture_scalar_outputs = True

# Replace torch.arange with torch.linspace
# Original: mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
mask = (torch.linspace(0, 511, 512) < 8).unsqueeze(0).cuda()

# Replace torch.randn with torch.nn.functional.max_pool2d
# Original: hidden = torch.randn((1, 512, 4096)).cuda()
# We create a dummy input tensor and apply max_pool2d to generate the 'hidden' tensor.
# The shape (1, 512, 4096) is treated as (1, 512, 64, 64) for the pooling operation.
dummy_input = torch.ones(1, 512, 64, 64).cuda()
hidden = F.max_pool2d(dummy_input, kernel_size=1).view(1, 512, 4096)

result = fn(mask, hidden)

# Verify the result shape matches the expected slice length (8)
assert result.shape == (1, 8, 3072)