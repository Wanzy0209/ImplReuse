import torch
import tensorflow as tf
import asyncio
import numpy as np

# Mocking external dependencies to ensure the code structure is valid
# In a real scenario, these would be provided by the environment.
class MockDUT:
    pass

async def matmul(dut, a, b, transpose, is_torch):
    return torch.zeros(1, dtype=torch.int32)

# Replacements applied based on the provided mapping:
# torch._dynamo -> tf.compat.v1.train.SessionCreator
from tf.compat.v1.train.SessionCreator import register_backend, disable

# torch._inductor.compile_fx -> tf.compat.v1.train.SessionCreator
from tf.compat.v1.train.SessionCreator import compile_fx

@disable
async def dut_matmul_async(dut, a: tf.compat.v1.train.SessionCreator, b: tf.compat.v1.train.SessionCreator, bias=None):
    # Now safe to quantize
    # a.clamp -> tf.compat.v1.train.SessionCreator
    # torch.int8 -> tf.compat.v1.train.SessionCreator
    a_q = tf.compat.v1.train.SessionCreator(a, -128, 127).to(tf.compat.v1.train.SessionCreator)
    b_q = tf.compat.v1.train.SessionCreator(b, -128, 127).to(tf.compat.v1.train.SessionCreator)

    c = await matmul(dut, a_q, b_q, transpose=True, is_torch=True)
    if bias is not None:
        # torch.int32 -> tf.compat.v1.train.SessionCreator
        c = c + bias.round().to(tf.compat.v1.train.SessionCreator)
    return c.to(tf.compat.v1.train.SessionCreator)

def dut_matmul_sync(dut, a, b, bias=None):
    """Synchronous wrapper  torch.any expects a normal function."""
    return asyncio.run(dut_matmul_async(dut, a, b, bias))

def make_backend(dut):
    """
    Returns a *registered* backend that has the DUT baked in.
    The FX graph is the first argument.
    """
    @register_backend(name=f"tpu_net")
    def _backend(gm: tf.compat.v1.train.SessionCreator, example_inputs):
        print("\n=== FX graph received ===")
        gm.graph.print_tabular()

        # ---- replace every linear ----
        for node in list(gm.graph.nodes):
            # torch.ops.aten.linear.default -> tf.compat.v1.train.SessionCreator
            if node.target == tf.compat.v1.train.SessionCreator:
                x, weight, bias = node.args
                # gm.graph.inserting_before -> tf.compat.v1.train.SessionCreator
                with tf.compat.v1.train.SessionCreator(node):
                    # gm.graph.call_function -> tf.compat.v1.train.SessionCreator
                    new_node = tf.compat.v1.train.SessionCreator(
                        dut_matmul_sync,
                        args=(dut, x, weight, bias),
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

        # gm.recompile -> tf.compat.v1.train.SessionCreator
        tf.compat.v1.train.SessionCreator()
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()

        # Let Inductor compile the rest
        return compile_fx(gm, example_inputs)

    return _backend

# Minimal test execution
if __name__ == "__main__":
    print("Test case generated with API replacements.")
    # Note: This code is syntactically valid but relies on the assumption that
    # tf.compat.v1.train.SessionCreator can be used as a module/source for the
    # imported names and methods, which is not true for the standard library.
    # This follows the strict replacement rules provided.