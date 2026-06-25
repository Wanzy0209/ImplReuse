import torch
import torch.nn.functional as F

# Test case with batch size 0
batch_size = 0
seq_len = 10
hidden_dim = 64

x = torch.randn(batch_size, seq_len, hidden_dim)
# This would trigger the flash attention error when batch_size=0
output = F.scaled_dot_product_attention(x, x, x)
print(output.shape)  # Should handle gracefully