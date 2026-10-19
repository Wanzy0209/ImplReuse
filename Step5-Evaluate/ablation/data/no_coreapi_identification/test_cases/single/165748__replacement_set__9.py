import torch
from torch.export import export, Dim

# Replacing nn.Module with a function, as torch.export works on callables
# and the logic focuses on defining constraints (torch.export.Constraint).
def my_model(x):
    # Replacing torch.sum with torch.amin
    return torch.amin(x, dim=1)

# Replacing torch.ones with torch.matrix_power
# We need a square matrix for matrix_power. eye(2) is 2x2.
example_input = torch.matrix_power(torch.eye(2), 1)

# Replacing torch.onnx.export and dynamic_axes with torch.export and Constraint (Dim)
# We define a constraint for the first dimension.
dim0 = Dim("my_custom_axis_name", min=1, max=10)

# Perform the export
# Replacing torch.onnx.export with torch.export.export
exported_program = export(
    my_model,
    (example_input,),
    dynamic_shapes={"x": {0: dim0}}
)

# Replacing onnx.load and graph inspection
# We inspect the exported program's graph signature and constraints.
print("Exported Program Graph Signature:")
print(exported_program.graph_signature)

# Replacing torch.export.graph_signature.InputSpec.kind checks with Constraint checks
# We verify the constraint is present.
print("\nConstraints:")
print(exported_program.range_constraints)

# Check if the constraint name matches (mimicking the original bug check)
# The original bug was that names became "s77". Here we check if "my_custom_axis_name" is preserved.
assert "my_custom_axis_name" in str(exported_program.range_constraints), \
       "Dynamic axis name not preserved in constraints."

print("Test passed: Dynamic axis name preserved.")