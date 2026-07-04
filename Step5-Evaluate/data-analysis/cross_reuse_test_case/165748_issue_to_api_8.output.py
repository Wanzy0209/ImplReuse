import torch
import sys

# torch.export is available in PyTorch 2.1+
# We handle the import error to prevent crashes on older versions.
try:
    from torch.export import export, Dim
except ImportError:
    print("Skipping test: 'torch.export' module not found. This test requires PyTorch 2.1+.")
    sys.exit(0)

class SumModule(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)

def test_torch_export_dynamic_axes_names():
    # Define a dynamic dimension (Constraint) with a custom name.
    # In torch.export, dynamic axes are defined using Dim objects (which are Constraints).
    # This corresponds to the 'dynamic_axes' argument in the original bug report.
    custom_axis = Dim("my_custom_axis_name", min=1, max=10)

    # Export the model using torch.export.export
    # We use the similar API (torch.export) as a candidate for reuse of the export logic.
    exported_program = export(
        SumModule(),
        args=(torch.ones(2, 2),),
        dynamic_shapes={"x": {0: custom_axis}},
    )

    # Verify the dynamic shape name is preserved in the exported program.
    # The original bug showed names being replaced by serial numbers (e.g., s77).
    # We check that our custom name exists in the exported program's constraints.
    found_name = False
    for constraint in exported_program.range_constraints.values():
        # Dim objects (Constraints) in torch.export have a name attribute.
        if hasattr(constraint, 'name') and constraint.name == "my_custom_axis_name":
            found_name = True
            break

    assert found_name, (
        f"Custom dynamic axis name 'my_custom_axis_name' was not preserved. "
        f"Found constraints: {exported_program.range_constraints}"
    )

if __name__ == "__main__":
    test_torch_export_dynamic_axes_names()
    print("Test passed: Dynamic axis name preserved in torch.export.")