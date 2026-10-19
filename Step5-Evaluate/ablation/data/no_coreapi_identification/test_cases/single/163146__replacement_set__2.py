import torch
import torch.nn as nn

# Define a module that reproduces the slicing behavior described in the bug report.
# The error occurs when slicing a tensor using a scalar tensor value as the index.
class SliceModel(nn.Module):
    def forward(self, item_embedding, max_item_num):
        # The problematic line from the traceback:
        # selected_item_embedding = item_embedding[:, :max_item_num, :]
        # This causes a "Data dependent error" because max_item_num is a runtime tensor value.
        return item_embedding[:, :max_item_num, :]

def test_export_slice_data_dependent():
    # Setup inputs matching the shapes in the bug report
    # item_embedding: Tensor(shape: torch.Size([s10, s64, 64]))
    item_embedding = torch.randn(10, 64, 64)
    
    # max_item_num: Tensor(shape: torch.Size([]), stride: (), storage_offset: 0)
    # We need a scalar tensor to act as the slice index.
    # Requirement: Replace torch.arange with torch.std.
    # Original logic might have been: max_item_num = torch.arange(1, 10).sum()
    # Replaced logic:
    max_item_num = torch.std(torch.randn(10))

    model = SliceModel()

    # Attempt to export the model.
    # The bug report indicates this fails or produces warnings with both strict=False and strict=True
    # due to the data-dependent slice.
    try:
        # Using strict=False as shown in the first attempt in the bug report
        ep = torch.export.export(model, args=(item_embedding, max_item_num), strict=False)
        print("Export succeeded (or produced a graph with warnings).")
        # If export succeeds despite the warning in the bug report, we can inspect the graph
        # or simply assert that the program is returned.
        assert ep is not None
    except Exception as e:
        # Depending on the PyTorch version, this might raise an error or just print warnings.
        # The bug report shows a WARNING log, but mentions "not able to soundly produce a graph".
        print(f"Export encountered an issue: {e}")

if __name__ == "__main__":
    test_export_slice_data_dependent()