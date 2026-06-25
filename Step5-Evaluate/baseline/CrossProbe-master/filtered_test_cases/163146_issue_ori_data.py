import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, item_embedding, max_item_num):
        # This line causes the data-dependent error
        selected_item_embedding = item_embedding[:, :max_item_num, :]
        return selected_item_embedding

# Reproduce the error
model = Mlp()
item_embedding = torch.randn(10, 64, 64)
max_item_num = torch.tensor(200)  # Dynamic value causing export issue

# Try to export - this will fail with data-dependent error
try:
    exported = torch.export.export(model, (item_embedding, max_item_num), strict=True)
except Exception as e:
    print(f"Export failed: {e}")