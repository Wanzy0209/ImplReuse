import torch
import torch._dynamo
import torch.nn as nn
import asyncio
import unittest

# --- Similar API Pattern: torch.backends.mkldnn.is_available ---
# We mimic the structure of torch.backends.mkldnn.is_available to check 
# for the presence of our custom backend prerequisites.
_has_custom_hardware = True

def is_custom_backend_available():
    r"""Return whether the custom TPU backend is available."""
    return _has_custom_hardware

# --- Bug Reproduction Logic ---

# Mocking the external hardware dependency to make the test runnable
async def mock_hardware_matmul(dut, a, b, transpose=False, is_torch=False):
    # Simulate the quantization and matmul logic from the bug report
    a_q = a.clamp(-128, 127).to(torch.int8)
    b_q = b.clamp(-128, 127).to(torch.int8)
    # Simple mock computation returning int32
    return torch.matmul(a_q.to(torch.float32), b_q.t().to(torch.float32)).to(torch.int32)

@torch._dynamo.disable
async def dut_matmul_async(dut, a: torch.Tensor, b: torch.Tensor, bias=None):
    # Reproduce the async logic from the issue
    c = await mock_hardware_matmul(dut, a, b, transpose=True, is_torch=True)
    if bias is not None:
        c = c + bias.round().to(torch.int32)
    return c.to(torch.int32)

def dut_matmul_sync(dut, a, b, bias=None):
    """Synchronous wrapper  torch.compile expects a normal function."""
    return asyncio.run(dut_matmul_async(dut, a, b, bias))

def make_backend(dut):
    """
    Returns a *registered* backend that has the DUT baked in.
    The FX graph is the first argument.
    """
    @torch._dynamo.register_backend(name="tpu_net")
    def _backend(gm: torch.fx.GraphModule, example_inputs):
        # ---- replace every linear (Graph manipulation from the bug) ----
        for node in list(gm.graph.nodes):
            if node.target == torch.ops.aten.linear.default:
                x, weight, bias = node.args
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        dut_matmul_sync,
                        args=(dut, x, weight, bias),
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

        gm.recompile()

        # Let Inductor compile the rest (or fallback if not available)
        try:
            from torch._inductor.compile_fx import compile_fx
            return compile_fx(gm, example_inputs)
        except ImportError:
            return gm

    return _backend

class TestCustomBackendGraphBreak(unittest.TestCase):
    def test_backend_registration_and_compile(self):
        # Leverage the similar API pattern to guard the test execution
        if not is_custom_backend_available():
            self.skipTest("Custom backend hardware not available")

        # Setup the custom backend
        dut = "simulated_chip"
        make_backend(dut)

        # Define a simple model to trigger the graph path
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)

        model = SimpleLinear()
        
        # Compile with the custom backend
        # This reproduces the scenario where the graph break might occur
        try:
            compiled_model = torch.compile(model, backend="tpu_net")
            input_data = torch.randn(2, 10)
            output = compiled_model(input_data)
            
            # Basic assertion to ensure execution completed without graph break errors
            self.assertEqual(output.shape, (2, 5))
        except Exception as e:
            self.fail(f"Backend compilation failed with error: {e}")

if __name__ == "__main__":
    unittest.main()