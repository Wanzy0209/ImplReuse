import torch
import torch.library

def test_export_with_registered_vmap():
    """
    Test case to verify torch.library.register_vmap compatibility with torch.export.export.
    Based on Issue 161563: "Current active mode not registered" when exporting vmap.
    """

    # 1. Define a custom operator using the Library class
    # This approach is used to avoid the AttributeError in the functional API
    # (torch.library.impl) present in some PyTorch versions.
    lib = torch.library.Library("test_ns", "DEF")
    lib.define("custom_op(Tensor x) -> Tensor")

    # 2. Implement the operator for CPU
    def custom_op_impl(x):
        return x * 2
    lib.impl("custom_op", custom_op_impl, "CPU")

    # 3. Register a vmap implementation for the operator
    # This is the core of the interaction that caused the bug in the original report.
    @torch.library.register_vmap("test_ns::custom_op")
    def custom_op_vmap(info, in_dims, x):
        # For this simple op, the vmap logic is straightforward.
        # We apply the operation and propagate the batch dimension.
        return x * 2, in_dims[0]

    # 4. Define a function to export that uses the custom op
    def fn(x):
        return torch.ops.test_ns.custom_op(x)

    # 5. Attempt to export the function
    # This mimics the original bug scenario where torch.export.export
    # failed on a model relying on vmap-registered ops.
    example_inputs = (torch.randn(2, 3),)
    
    try:
        ep = torch.export.export(fn, example_inputs)
        print("Export successful.")
        # Verify the exported graph contains the custom op
        graph_str = str(ep.graph)
        assert "test_ns::custom_op" in graph_str
        print("Test passed: Custom op with registered vmap exported successfully.")
    except AssertionError as e:
        if "Current active mode" in str(e):
            print(f"Bug reproduced: {e}")
            raise
        else:
            raise

if __name__ == "__main__":
    test_export_with_registered_vmap()