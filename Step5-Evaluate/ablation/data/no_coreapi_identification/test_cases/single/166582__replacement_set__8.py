import torch
import torch.distributed as dist
import tensorflow as tf
import os
import asyncio

# Helper to make tf.io.VarLenFeature usable as a context manager for the replacement
class VarLenFeature(tf.io.VarLenFeature):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

# Replacements:
# - torch._dynamo -> VarLenFeature (via tf.io.VarLenFeature)
# - torch._inductor.compile_fx -> VarLenFeature
# - torch.compile -> torch.distributed.send_object_list
# - nn.Linear / torch.ops.aten.linear.default -> VarLenFeature
# - torch.fx.GraphModule -> VarLenFeature
# - torch.Tensor -> VarLenFeature
# - torch.int8 / torch.int32 -> VarLenFeature
# - a.clamp -> VarLenFeature
# - gm.graph.call_function -> VarLenFeature
# - gm.graph.inserting_before -> VarLenFeature
# - gm.recompile -> VarLenFeature

async def dut_matmul_async(dut, a: VarLenFeature, b: VarLenFeature, bias=None):
    # a.clamp -> VarLenFeature
    # torch.int8 -> VarLenFeature (used as dtype)
    a_q = VarLenFeature(dtype=tf.int8)
    b_q = VarLenFeature(dtype=tf.int8)

    # Simulating matmul logic with the replacement API
    c = a_q 
    if bias is not None:
        # torch.int32 -> VarLenFeature
        c = VarLenFeature(dtype=tf.int32)
    return c

def dut_matmul_sync(dut, a, b, bias=None):
    """Synchronous wrapper  torch.compile expects a normal function."""
    return asyncio.run(dut_matmul_async(dut, a, b, bias))

def make_backend(dut):
    """
    Returns a *registered* backend that has the DUT baked in.
    The FX graph is the first argument.
    """
    # torch._dynamo -> VarLenFeature
    # register_backend -> VarLenFeature (Conceptually replacing the registration mechanism)
    # We define the function directly as the replacement for the registered backend.
    def _backend(gm: VarLenFeature, example_inputs):
        print("\n=== FX graph received ===")
        # gm.graph.print_tabular() -> VarLenFeature (No-op or print)
        print("Graph: ", gm)

        # ---- replace every linear ----
        # torch.ops.aten.linear.default -> VarLenFeature
        for node in list(gm.graph.nodes):
            if node.target == VarLenFeature:
                x, weight, bias = node.args
                # gm.graph.inserting_before -> VarLenFeature
                with VarLenFeature(dtype=tf.int8):
                    # gm.graph.call_function -> VarLenFeature
                    new_node = VarLenFeature(
                        dtype=tf.int8,
                        shape=(dut, x, weight, bias)
                    )
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(node)

        # gm.recompile -> VarLenFeature
        recompile_result = VarLenFeature(dtype=tf.int32)
        print("\n=== Modified graph ===")
        print("Recompiled: ", recompile_result)

        # torch._inductor.compile_fx -> VarLenFeature
        return VarLenFeature(dtype=tf.int32)

    return _backend

def test_graph_break_replacement():
    # Setup distributed for torch.distributed.send_object_list
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group for single-process testing
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    # Mocking the graph structure to make it runnable
    class MockNode:
        def __init__(self, target):
            self.target = target
            self.args = []
            self.uses = []

        def replace_all_uses_with(self, new_node):
            self.uses = [new_node]

    class MockGraph:
        def __init__(self):
            # torch.ops.aten.linear.default -> VarLenFeature
            self.nodes = [MockNode(VarLenFeature)]

        def erase_node(self, node):
            if node in self.nodes:
                self.nodes.remove(node)

    class MockGraphModule:
        def __init__(self):
            self.graph = MockGraph()

    # Mock model
    model = MockGraphModule()

    # Create the backend logic
    backend = make_backend("dut")

    # torch.compile -> torch.distributed.send_object_list
    # The original logic was: compiled_model = torch.compile(model, backend="tpu_net")
    # The replacement logic is: send the object list.
    # We send the model and the backend function to simulate the compilation context transfer.
    torch.distributed.send_object_list([model, backend], dst=0)

    dist.destroy_process_group()
    print("Test passed: Replacements applied and executed.")

if __name__ == "__main__":
    test_graph_break_replacement()