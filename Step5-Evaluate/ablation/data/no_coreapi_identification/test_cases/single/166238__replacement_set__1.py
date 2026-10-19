import torch
import torch.nn.functional as F

def test_hardtanh_compile():
    """
    Test case generated based on Issue 166238 (Dynamo regression with collections.defaultdict).
    This test replaces the original failing API (collections.defaultdict) with the similar API
    (torch.nn.functional.hardtanh_) to verify correct behavior under torch.compile.
    """
    
    # Define a function that uses the replacement API: torch.nn.functional.hardtanh_
    # Original logic used collections.defaultdict which caused torch._dynamo.exc.Unsupported
    def fn(x):
        # Replacement: collections.defaultdict -> torch.nn.functional.hardtanh_
        # Instead of creating a defaultdict, we apply the hardtanh_ operation.
        return F.hardtanh_(x)

    input_tensor = torch.randn(3, 3)

    # Original API Under Test: torch.compile
    compiled_fn = torch.compile(fn)

    # Execute the compiled function.
    # Replacement: torch._dynamo.exc.Unsupported -> torch.nn.functional.hardtanh_
    # In the original bug, this raised an Unsupported error. 
    # Here, we expect the function to execute successfully, effectively replacing the error path 
    # with the successful execution of the target function.
    output = compiled_fn(input_tensor)

    # Verify the result matches eager execution
    expected = F.hardtanh_(input_tensor.clone())
    assert torch.equal(output, expected), "Output of compiled function does not match eager execution"

    # Replacement: tx.output.graph.nodes.call_function -> torch.nn.functional.hardtanh_
    # The original error context involved inspecting graph nodes for call_function.
    # We verify that the operation was performed correctly, implying the graph 
    # successfully handled the call to hardtanh_.
    print("Test passed: torch.compile successfully handled torch.nn.functional.hardtanh_")

if __name__ == "__main__":
    test_hardtanh_compile()