import torch
from torch import Tensor
from torch.library import Library, impl, register_vmap

def test_register_vmap_dynamic_slice():
    """
    Test case for torch.library.register_vmap inspired by Issue #163146.
    
    The original issue involves a data-dependent error during torch.export.export
    when slicing a tensor with a dynamic value (item_embedding[:, :max_item_num, :]).
    
    This test verifies that we can correctly register a vmap implementation
    for a custom operator that performs slicing, ensuring that the batching
    logic is handled correctly via torch.library.register_vmap.
    """

    # 1. Define a custom library and operator
    lib = Library("test_slice_ops", "DEF")
    lib.define("dynamic_slice(Tensor x, int n) -> Tensor")

    # 2. Implement the eager mode (fallback) implementation
    @impl(lib, "dynamic_slice", "CompositeExplicitAutograd")
    def dynamic_slice_impl(x: Tensor, n: int) -> Tensor:
        # Mimics the slicing logic from the bug report: x[:, :n, :]
        # We assume n is an integer here for the eager execution context.
        return x[:, :n, :]

    # 3. Register the vmap implementation using torch.library.register_vmap
    @register_vmap(lib, "dynamic_slice")
    def dynamic_slice_vmap(info, in_dims, x, n):
        """
        vmap implementation for dynamic_slice.
        
        Args:
            info: VmapInfo object containing metadata.
            in_dims: Tuple specifying the batch dimension for each input.
                     x_bdim is the batch dim for x, n_bdim is for n.
            x: The input tensor.
            n: The slice limit (integer).
        """
        x_bdim, n_bdim = in_dims

        # If n is not batched (None), we simply move the batch dimension of x
        # to the front (dim 0), apply the slice, and return the result with batch dim 0.
        if n_bdim is None:
            if x_bdim is None:
                # No batching
                return dynamic_slice_impl(x, n), None
            else:
                # Batch x, but n is constant across the batch
                # Move the batch dimension to 0 for processing
                x = torch.movedim(x, x_bdim, 0)
                result = dynamic_slice_impl(x, n)
                # The result has the batch dimension at 0
                return result, 0
        
        # If n were batched, we would need to handle ragged tensors or loop,
        # which is complex and not the focus of this basic API test.
        # We raise an error or handle simply if needed, but here we assume n is static.
        raise ValueError("Batching over the slice limit 'n' is not supported in this implementation.")

    # 4. Verification / Test Execution
    
    # Create a sample input tensor
    # Shape: (Batch=10, Seq=20, Dim=5)
    x = torch.randn(10, 20, 5)
    n = 5 # Slice limit

    # Test eager execution
    expected = torch.ops.test_slice_ops.dynamic_slice(x, n)
    assert expected.shape == torch.Size([10, 5, 5]), "Eager execution shape mismatch"

    # Test vmap execution
    # We want to map over the batch dimension (dim 0) of x.
    # The operator slices dim 1.
    # Input to vmap: (10, 20, 5). 
    # Inside vmap (per batch): (20, 5) -> slice -> (5, 5).
    # Output: (10, 5, 5).
    
    vmapped_func = torch.vmap(torch.ops.test_slice_ops.dynamic_slice, in_dims=(0, None))
    actual = vmapped_func(x, n)

    assert actual.shape == torch.Size([10, 5, 5]), "Vmap execution shape mismatch"
    assert torch.allclose(actual, expected), "Vmap execution value mismatch"
    
    print("Test passed: torch.library.register_vmap works correctly for slicing operations.")

if __name__ == "__main__":
    test_register_vmap_dynamic_slice()