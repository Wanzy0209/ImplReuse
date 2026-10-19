import torch
from torch import export

class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item_embedding, scores):
        # The bug report describes a data-dependent error during export when slicing.
        # The slicing index `max_item_num` is derived from input data.
        #
        # Requirement: Replace torch.arange with torch.quantile.
        # Original logic (hypothetical) might have been:
        #   max_item_num = torch.arange(scores.size(0)).max().item()
        #
        # Replaced logic:
        # We use torch.quantile to determine the slice index based on the distribution of 'scores'.
        # This ensures the slice index is data-dependent.
        max_item_num = torch.quantile(scores, 0.5)

        # The error log shows max_item_num is a Tensor.
        # We cast to int to allow slicing (slicing requires integers).
        max_item_num = max_item_num.to(torch.int)

        # This is the problematic line from the bug report:
        # selected_item_embedding = item_embedding[:, :max_item_num, :]
        # This causes a "Data dependent error" in torch.export because the shape of the output
        # depends on the value of max_item_num, which depends on the data in 'scores'.
        selected_item_embedding = item_embedding[:, :max_item_num, :]

        return selected_item_embedding

def test_export_slice_with_quantile():
    # Setup inputs
    # item_embedding shape: [batch, seq_len, dim]
    # scores shape: [seq_len]
    batch_size = 2
    seq_len = 10
    dim = 64

    item_embedding = torch.randn(batch_size, seq_len, dim)
    scores = torch.rand(seq_len)

    model = Mlp()

    # 1. Test Eager Mode
    print("Testing Eager Mode...")
    try:
        out_eager = model(item_embedding, scores)
        print(f"Eager Mode Success. Output shape: {out_eager.shape}")
    except Exception as e:
        print(f"Eager Mode Failed: {e}")
        return

    # 2. Test torch.export.export
    # The bug report indicates this fails with "Data dependent error"
    # because the slice index depends on the value of 'scores'.
    print("\nTesting torch.export.export...")
    try:
        # Attempting export with strict=False (as in bug report)
        ep = export.export(model, args=(item_embedding, scores), strict=False)
        print("Export Success (strict=False). Graph generated.")
        # If this succeeds, the bug might be fixed or the specific condition isn't met.
        # However, based on the issue, we expect a failure or a constraint warning.
    except Exception as e:
        print(f"Export Failed: {e}")
        # Check for the specific error mentioned in the bug report
        if "Data dependent error" in str(e):
            print("Caught expected 'Data dependent error'.")
        else:
            print("Caught a different error.")

if __name__ == "__main__":
    test_export_slice_with_quantile()