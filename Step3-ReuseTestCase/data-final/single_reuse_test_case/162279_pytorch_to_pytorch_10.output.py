import torch
import torch.nn as nn
import torch.library

# Define a custom operator using torch.library.define to mimic aten::any.dims
# This tests the interaction between custom operator definitions and the export process.
torch.library.define("custom_ns::any_dims", "(Tensor x, int[]? dims, bool keepdim) -> Tensor")

@torch.library.impl("custom_ns::any_dims", "CPU")
def any_dims_cpu(x, dims, keepdim):
    return torch.any(x, dim=dims, keepdim=keepdim)

# Define a fake implementation for export/metas
@torch.library.impl_abstract("custom_ns::any_dims")
def any_dims_abstract(x, dims, keepdim):
    # For the purpose of this test, we assume the shape logic follows torch.any
    # In a real scenario, this would need to be accurate for shape inference
    if dims is None:
        dims = list(range(x.dim()))
    elif isinstance(dims, list) and len(dims) == 0:
        # If dims is empty list, reduce over all dims (scalar output)
        dims = list(range(x.dim()))
    
    shape = []
    if keepdim:
        for i in range(x.dim()):
            shape.append(1 if i in dims else x.size(i))
    else:
        for i in range(x.dim()):
            if i not in dims:
                shape.append(x.size(i))
    
    if not shape:
        shape = [1]
        
    return x.new_empty(shape)

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def forward(self, x):
        # Use the custom operator defined via torch.library.define
        y = torch.ops.custom_ns.any_dims(x, [], False)
        return y

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def forward(self, x):
        # Use the custom operator defined via torch.library.define
        y = torch.ops.custom_ns.any_dims(x, None, False)
        return y

def process(model, x):
    print(f"Testing model: {model.__class__.__name__}")
    
    # Run eager mode
    eager_output = model(x)
    print(f"Eager output shape: {eager_output.shape}")
    
    # Export the model
    print("Exporting model...")
    exported_program = torch.export.export(model, (x,))
    
    # Verify the output shape of the exported graph matches eager mode
    # This checks if the export process correctly handles the custom op's shape
    exported_output = exported_program(x)
    print(f"Export output shape: {exported_output.shape}")
    
    assert eager_output.shape == exported_output.shape, \
        f"Shape mismatch: eager {eager_output.shape} vs export {exported_output.shape}"
    print("Test passed.\n")

if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    
    # Process the first model
    process(AnyDimsModelEmpty(), x)
    
    # Process the second model
    # The original bug report suggests that processing a second model with different
    # arguments (empty list vs None) might cause incorrect output shapes due to caching.
    process(AnyDimsModelNull(), x)