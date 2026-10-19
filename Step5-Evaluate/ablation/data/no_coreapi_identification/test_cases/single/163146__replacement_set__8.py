import torch
import torch.nn as nn
from torch.export import export

class Mlp(nn.Module):
    def forward(self, item_embedding, limit_vector):
        # The bug report involves slicing based on a dynamic value (max_item_num).
        # We use torch.norm to calculate this value, replacing torch.arange.
        # torch.norm returns a float tensor, which can be used for slicing (truncated to int).
        max_item_num = torch.norm(limit_vector)
        
        # This is the line causing the error in the bug report:
        # "Data dependent error... unable to evaluate the value of u1"
        selected_item_embedding = item_embedding[:, :max_item_num, :]
        return selected_item_embedding

def test_export_slice_with_norm():
    # Setup inputs based on the bug report's local variables
    # item_embedding: Tensor(shape: torch.Size([s10, s64, 64]))
    item_embedding = torch.randn(10, 64, 64)
    
    # A vector to calculate the norm from, acting as the source for the dynamic slice size
    limit_vector = torch.randn(10)

    model = Mlp()

    # Attempt to export with strict=True, as indicated in the bug report traceback
    # The bug report indicates this fails with a "Data dependent error"
    try:
        ep = export(model, (item_embedding, limit_vector), strict=True)
        print("Export succeeded.")
    except Exception as e:
        print(f"Export failed with error: {e}")
        # In a strict regression test, we might assert the error message matches the bug report
        # assert "Data dependent error" in str(e) or "Unable to evaluate" in str(e)

if __name__ == "__main__":
    test_export_slice_with_norm()