import torch
from torch.export import Dim, export

def test_reshape_copy_unbacked_semantics():
    """
    Test case for Issue 162110: unbacked semantics for _reshape_copy not defined.
    
    This test reproduces the logic of the original bug report by attempting to 
    decompose `torch.ops.aten.view.default` into `torch.ops.aten._reshape_copy.default`
    within a `torch.export` context that utilizes dynamic shapes.
    
    The test leverages the pattern of defining a wrapper function (similar to the 
    provided `tf.compat.v1.train.global_step` example) to handle the operator call.
    """
    # Define a dynamic dimension for the export context
    seq_len = Dim("seq_len", min=1, max=128)

    # Define the decomposition wrapper, mirroring the pattern of the similar API
    # (a simple function wrapping a specific operation).
    def view_decomposition(x: torch.Tensor, size: list[torch.SymInt]) -> torch.Tensor:
        return torch.ops.aten._reshape_copy.default(x, size)

    # Define a simple module that uses `view`, which triggers the decomposition path
    class ViewModel(torch.nn.Module):
        def forward(self, x):
            # Reshape (B, S) to (B, S, 1) to ensure view is used
            return x.view(x.size(0), x.size(1), 1)

    model = ViewModel()
    
    # Create example inputs
    # Shape (1, 12) where 12 is the dynamic dimension
    inputs = torch.randn(1, 12)
    
    # Specify dynamic shapes
    dynamic_shapes = ({1: seq_len},)

    try:
        # Export the model with dynamic shapes
        ep = export(
            model,
            args=(inputs,),
            dynamic_shapes=dynamic_shapes,
            strict=False
        )
        
        # Set up the decomposition table
        decomp_table = export.default_decompositions()
        # Replace the default view op with our custom _reshape_copy wrapper
        decomp_table[torch.ops.aten.view.default] = view_decomposition

        # Run decompositions
        # In the buggy version, this would fail due to unbacked semantics in _reshape_copy
        after_decomp = ep.run_decompositions(decomp_table=decomp_table)
        
        # Verify that the graph contains the expected operator
        graph_module = after_decomp
        found_reshape_copy = any(
            node.target == torch.ops.aten._reshape_copy.default 
            for node in graph_module.graph.nodes
        )
        
        assert found_reshape_copy, "Decomposition failed to insert _reshape_copy.default"
        print("Test Passed: _reshape_copy handles dynamic shapes correctly.")

    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_reshape_copy_unbacked_semantics()