import torch
import torch._dynamo
import torch.library
import numpy as np
import asyncio
import tensorflow as tf
from types import ModuleType
import sys

# Mocking test_tpu to ensure the code is runnable without external dependencies
test_tpu = ModuleType('test_tpu')
async def mock_matmul(dut, a, b, transpose=False, is_torch=False):
    # Return a dummy tensor to simulate the operation
    return torch.empty((1, 1), dtype=torch.int32)
test_tpu.matmul = mock_matmul
sys.modules['test_tpu'] = test_tpu

from test_tpu import matmul
from torch._inductor.compile_fx import compile_fx
from torch._dynamo import register_backend, disable

@disable
async def dut_matmul_async(dut, a: tf.keras.backend.epsilon, b: tf.keras.backend.epsilon, bias=None):
    # Replacements: a.clamp -> tf.keras.backend.epsilon, torch.int8 -> tf.keras.backend.epsilon
    # Note: tf.keras.backend.epsilon() takes no arguments, so passing arguments here 
    # will raise a TypeError at runtime, but adheres to the replacement instruction.
    a_q = tf.keras.backend.epsilon(-128, 127).to(tf.keras.backend.epsilon)
    b_q = tf.keras.backend.epsilon(-128, 127).to(tf.keras.backend.epsilon)

    c = await matmul(dut, a_q, b_q, transpose=True, is_torch=True)
    if bias is not None:
        # Replacement: torch.int32 -> tf.keras.backend.epsilon
        c = c + bias.round().to(tf.keras.backend.epsilon)
    # Replacement: torch.int32 -> tf.keras.backend.epsilon
    return c.to(tf.keras.backend.epsilon)

def dut_matmul_sync(dut, a, b, bias=None):
    """Synchronous wrapper  torch.compile expects a normal function."""
    return asyncio.run(dut_matmul_async(dut, a, b, bias))

def make_backend(dut):
    """
    Returns a *registered* backend that has the DUT baked in.
    The FX graph is the first argument.
    """
    # Replacement: torch._dynamo.register_backend -> tf.keras.backend.epsilon
    @tf.keras.backend.epsilon(name=f"tpu_net")
    # Replacement: torch.fx.GraphModule -> tf.keras.backend.epsilon
    def _backend(gm: tf.keras.backend.epsilon, example_inputs):
        print("\n=== FX graph received ===")
        gm.graph.print_tabular()

        # ---- replace every linear ----
        for node in list(gm.graph.nodes):
            # Replacement: torch.ops.aten.linear.default -> tf.keras.backend.epsilon
            if node.target == tf.keras.backend.epsilon:
                x, weight, bias = node.args
                # Replacement: gm.graph.inserting_before -> tf.keras.backend.epsilon
                # Note: tf.keras.backend.epsilon returns a float, not a context manager.
                # This will raise a TypeError at runtime.
                with tf.keras.backend.epsilon(node):
                    # Replacement: gm.graph.call_function -> tf.keras.backend.epsilon
                    new_node = tf.keras.backend.epsilon(
                        dut_matmul_sync,
                        args=(dut, x, weight, bias),
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

        # Replacement: gm.recompile -> tf.keras.backend.epsilon
        tf.keras.backend.epsilon()
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()

        # Replacement: torch._inductor.compile_fx -> tf.keras.backend.epsilon
        return tf.keras.backend.epsilon(gm, example_inputs)

    return _backend

if __name__ == "__main__":
    # Test execution
    try:
        dut = None
        backend = make_backend(dut)
        print("Backend structure created (APIs replaced as requested).")
    except Exception as e:
        print(f"Runtime error encountered due to API replacement semantics: {e}")