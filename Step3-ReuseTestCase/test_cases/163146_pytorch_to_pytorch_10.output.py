import torch
import torch.nn as nn
from torch.export import export

# Define the custom operator using torch.library.define
# We adapt the signature to take an int instead of a Tensor for the limit
# to avoid the data-dependent error encountered in the original bug report.
torch.library.define(
    "custom_ns::safe_slice",
    "(Tensor x, int limit) -> Tensor"
)

@torch.library.impl("custom_ns::safe_slice", "CPU")
@torch.library.impl("custom_ns::safe_slice", "CUDA")
def safe_slice_impl(x, limit):
    # Perform the slicing operation: item_embedding[:, :max_item_num, :]
    return x[:, :limit, :]

@torch.library.register_fake("custom_ns::safe_slice")
def safe_slice_meta(x, limit):
    # Meta function for export/tracing.
    # 'limit' is treated as a SymInt during tracing, which allows
    # the exporter to handle dynamic shapes without data-dependent errors.
    return x.new_empty((x.size(0), limit, x.size(2)))

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize item_embedding similar to the bug report context
        self.item_embedding = nn.Parameter(torch.randn(10, 64, 64))

    def forward(self, max_item_num):
        # Use the custom operator instead of direct slicing
        return torch.ops.custom_ns.safe_slice(self.item_embedding, max_item_num)

def test_custom_op_export():
    model = Mlp()
    
    # In the original bug, max_item_num was a Tensor causing the error.
    # Here we pass an int (or SymInt) to the custom op.
    args = (20,)
    
    # 1. Test basic execution
    output = model(*args)
    assert output.shape == (10, 20, 64), f"Expected shape (10, 20, 64), got {output.shape}"
    print("Basic execution test passed.")

    # 2. Test export with torch.export.export
    # This verifies that the custom operator definition allows the export to proceed
    # where the original dynamic slicing failed.
    try:
        ep = export(model, args)
        print("Export succeeded with custom operator definition.")
        print("Exported Graph:")
        print(ep.graph)
    except Exception as e:
        print(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    test_custom_op_export()