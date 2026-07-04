import torch
from collections import namedtuple

# Attempt to use the new torch.library API
try:
    from torch.library import register_vmap, define
    
    # Define a custom operator to use with register_vmap
    def custom_op_meta(x, y):
        return torch.empty_like(x)

    def custom_op_impl(x, y):
        return x + y

    # Define the custom op
    define("test_ns::custom_add", "(Tensor, Tensor) -> Tensor", meta=custom_op_meta)
    from torch.ops import test_ns
    test_ns.custom_add.impl(custom_op_impl)

    # Register vmap implementation for the custom op
    def custom_add_vmap(info, in_dims, x, y):
        # For element-wise addition, we simply pass the tensors through.
        # The batching dimension is preserved.
        # in_dims is a tuple (dim_x, dim_y).
        # Assuming both inputs are batched on the same dimension for this test.
        return test_ns.custom_add(x, y), in_dims[0]

    register_vmap("test_ns::custom_add")(custom_add_vmap)
    
    # Use the custom op
    custom_op_func = test_ns.custom_add

except ImportError:
    # Fallback for environments where torch.library API is not available (e.g., older PyTorch versions)
    # We use a standard function which vmap can handle natively
    print("Warning: torch.library API not available. Using fallback function.")
    def custom_op_func(x, y):
        return x + y

# Test case adapted from the bug report
def test_namedtuple_with_vmap():
    Point = namedtuple('Point', 'x y')

    class M(torch.nn.Module):
        def forward(self, p):
            # The module unpacks the NamedTuple
            return custom_op_func(p.x, p.y)

    # Create a batched input (batch size 2, feature size 3)
    inp = Point(torch.ones(2, 3), torch.ones(2, 3))

    # Run vmap. This relies on the registered vmap rule for test_ns::custom_add
    # and the ability of vmap to handle the NamedTuple input structure.
    result = torch.vmap(M())(inp)

    # Verify result
    expected = torch.ones(2, 3) * 2
    assert torch.allclose(result, expected)
    print("Test passed.")

if __name__ == "__main__":
    test_namedtuple_with_vmap()