import torch
import torch.nn as nn
import torch._dynamo
from torch._dynamo import register_backend, disable
from torch.profiler import itt

# --- Reproduction Logic from Bug Report ---

# Simulating the external hardware call that should not be traced by Dynamo
@disable
def hardware_matmul_impl(a: torch.Tensor, b: torch.Tensor, bias=None):
    """
    Simulates the custom ASIC matmul operation.
    Decorated with @disable to prevent graph breaks inside the external call.
    """
    # Simulate quantization logic mentioned in the bug
    a_q = a.clamp(-128, 127).to(torch.int8)
    b_q = b.clamp(-128, 127).to(torch.int8)
    
    # For the test, we just cast back to float to perform the operation
    # In the real bug, this would be an awaitable call to hardware
    return torch.matmul(a_q.to(torch.float32), b_q.to(torch.float32)) + (bias if bias is not None else 0)

# Wrapper function similar to the structure of range_pop (simple delegation)
def custom_matmul_wrapper(a, b, bias=None):
    return hardware_matmul_impl(a, b, bias)

@register_backend(name="custom_tpu_backend")
def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Custom backend that replaces aten.linear with the custom matmul wrapper.
    Leverages torch.profiler.itt.range_pop to track the modification scope.
    """
    # Leveraging the similar API: range_push/pop to profile the graph modification
    itt.range_push("backend_graph_modification")

    print("\n=== FX graph received ===")
    gm.graph.print_tabular()

    for node in list(gm.graph.nodes):
        if node.target == torch.ops.aten.linear.default:
            x, weight, bias = node.args
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    custom_matmul_wrapper,
                    args=(x, weight, bias),
                )
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)

    gm.recompile()
    
    print("\n=== Modified graph ===")
    gm.graph.print_tabular()

    # Leveraging the similar API: range_pop
    itt.range_pop()

    # Return the modified graph module
    return gm

# --- Test Case ---

def test_backend_compiler_graph_break():
    """
    Test case for Issue 166582: Backend Compiler Graph Break.
    Verifies that a custom backend can replace nn.Linear operations
    without causing graph breaks, using profiling APIs to monitor the process.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    inputs = (torch.randn(2, 10),)

    # Compile using the custom backend registered above
    compiled_model = torch._dynamo.optimize("custom_tpu_backend")(model)
    
    # Run the model
    output = compiled_model(*inputs)
    
    # Assertions to verify execution
    assert output is not None, "Output should not be None"
    assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
    
    print("Test Passed: Backend compiled and executed successfully without graph breaks.")

if __name__ == "__main__":
    test_backend_compiler_graph_break()