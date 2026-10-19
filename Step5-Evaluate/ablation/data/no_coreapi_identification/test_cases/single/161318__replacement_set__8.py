import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True)
def fn(encoder_attention_mask, encoder_hidden_states):
    encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
    text_len = encoder_attention_mask.sum().item()
    encoder_hidden_states = encoder_hidden_states[:, :text_len]
    return encoder_hidden_states

torch._dynamo.config.capture_scalar_outputs = True

# Replacement 1: torch.arange -> torch.norm
# Original: mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
# We construct a tensor where the norm of the first 8 columns is < 8, and the rest is > 8.
# This mimics the behavior of arange(512) < 8.
base_mask_tensor = torch.zeros(512, 512).cuda()
base_mask_tensor[:, 8:] = 10.0  # Set values high for indices > 8
# Compute norm along dim 0 (columns). Shape becomes [512].
# First 8 norms are 0. Rest are > 8.
mask_vals = torch.norm(base_mask_tensor, dim=0)
mask = (mask_vals < 8).unsqueeze(0).cuda()

# Replacement 2: torch.randn -> torch.nn.functional.max_pool3d
# Original: hidden = torch.randn((1, 512, 4096)).cuda()
# We use max_pool3d. It requires a 5D input.
# We create a dummy 5D tensor, pool it, and reshape to 3D.
# Input shape (1, 512, 4, 32, 32) -> Pooled shape (1, 512, 2, 16, 16) -> Reshaped (1, 512, 512)
pool_input = torch.ones(1, 512, 4, 32, 32).cuda()
pooled = F.max_pool3d(pool_input, kernel_size=2)
hidden = pooled.reshape(1, 512, -1)

# Run the test
try:
    result = fn(mask, hidden)
    # Basic assertion to check if it ran and produced output
    assert result.shape[0] == 1
    assert result.shape[1] == 512
    # The slice length depends on the mask. In our construction, mask has 8 True values.
    # So text_len should be 8.
    assert result.shape[2] == 8
    print("Test passed successfully.")
except Exception as e:
    print(f"Test failed: {e}")
    raise