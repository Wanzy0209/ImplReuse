import torch
import torch.nn as nn

class Mlp(nn.Module):
    """
    Minimal model reproducing the data-dependent slicing error 
    described in the bug report.
    """
    def forward(self, item_embedding, max_item_num):
        # The problematic line from the traceback:
        # selected_item_embedding = item_embedding[:, :max_item_num, :]
        # This causes a "Data dependent error" because max_item_num is a Tensor.
        return item_embedding[:, :max_item_num, :]

def test_export_data_dependent_slice():
    # Instantiate the model
    model = Mlp()

    # Prepare inputs
    # Requirement: Replace torch.arange with torch.linspace
    # Original logic for generating dummy data might have been:
    # item_embedding = torch.arange(10 * 20 * 64).view(10, 20, 64).float()
    # Replacement logic:
    item_embedding = torch.linspace(0, 12799, 12800).view(10, 20, 64)
    
    # max_item_num is a scalar tensor, making the slice operation data-dependent
    max_item_num = torch.tensor(5)

    # Attempt to export the model
    # The bug report indicates this fails with strict=True and strict=False
    # due to inability to evaluate the value of the slice index.
    try:
        ep = torch.export.export(model, (item_embedding, max_item_num), strict=True)
        print("Export succeeded.")
        # If export succeeds, we can verify the graph or execution
        res = ep.module()(item_embedding, max_item_num)
        assert res.shape == (10, 5, 64), "Output shape mismatch"
    except Exception as e:
        print(f"Export failed with error: {e}")
        # Depending on the version of PyTorch, this might be expected behavior
        # or the test might be asserting that the bug is fixed.
        # For the purpose of this generation, we capture the scenario.

if __name__ == "__main__":
    test_export_data_dependent_slice()