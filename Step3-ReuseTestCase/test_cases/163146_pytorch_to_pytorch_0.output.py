import torch
import torch.nn as nn
from torch import export

class DynamicSliceModule(nn.Module):
    """
    Minimal module reproducing the data-dependent slicing issue.
    Based on the traceback: selected_item_embedding = item_embedding[:, :max_item_num, :]
    """
    def forward(self, item_embedding, max_item_num):
        # The slice stop index is a tensor value (max_item_num), 
        # which causes the data-dependent error during export.
        return item_embedding[:, :max_item_num, :]

def test_export_dynamic_slice():
    model = DynamicSliceModule()
    
    # Setup inputs matching the traceback context
    # item_embedding: Tensor(shape: torch.Size([s10, s64, 64]))
    item_embedding = torch.randn(10, 64, 64)
    
    # max_item_num: Tensor(shape: torch.Size([]))
    # This scalar tensor is used as the slice index, causing the issue.
    max_item_num = torch.tensor(32)

    print("Attempting to export model with dynamic slicing...")

    # The bug report indicates this fails with a "Data dependent error"
    # because torch.export cannot evaluate the value of the tensor index statically.
    try:
        ep = export(model, (item_embedding, max_item_num))
        print("Export succeeded.")
    except Exception as e:
        print(f"Export failed with error: {e}")
        # Re-raise to signal failure if the bug is present
        raise

if __name__ == "__main__":
    test_export_dynamic_slice()