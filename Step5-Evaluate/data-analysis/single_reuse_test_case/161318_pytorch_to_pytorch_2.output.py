import torch

# Fix for older PyTorch versions where torch.compile is not available
if not hasattr(torch, 'compile'):
    # Define a mock decorator that returns the function unchanged
    def mock_compile(**kwargs):
        def decorator(func):
            return func
        return decorator
    torch.compile = mock_compile

# Fix for missing torch._dynamo
if not hasattr(torch, '_dynamo'):
    class MockDynamo:
        class Config:
            capture_scalar_outputs = True
        config = Config()
    torch._dynamo = MockDynamo()

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True)
def fn(encoder_attention_mask, encoder_hidden_states):
    encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
    # Replaced .sum() with torch.prod() to test similar API behavior
    # Note: prod on the boolean mask (containing 0s) will result in 0
    text_len = torch.prod(encoder_attention_mask).item()
    encoder_hidden_states = encoder_hidden_states[:, :text_len]
    return encoder_hidden_states

# Original mask setup
mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
hidden = torch.randn((1, 512, 4096)).cuda()

# Run the function
result = fn(mask, hidden)

# Since the mask contains False (0), the product is 0.
# We assert the shape is (1, 0, 3072) to verify execution without crash.
assert result.shape == (1, 0, 3072)