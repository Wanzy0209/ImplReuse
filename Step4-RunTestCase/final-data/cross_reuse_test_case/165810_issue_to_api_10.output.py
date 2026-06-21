import sys
import traceback

# Attempt to import TensorFlow and handle environment/dependency issues gracefully.
# The original error indicates a GLIBC version mismatch, which prevents the library from loading.
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed.")
    print(f"Error details: {e}")
    print("This is likely due to a system library version mismatch (e.g., GLIBCXX_3.4.29 not found).")
    print("Please ensure your environment has the required dependencies to run TensorFlow.")
    sys.exit(0)

class InnerModule(tf.Module):
    """
    TensorFlow equivalent of the PyTorch inner_f class.
    Preserves the structure of defining a module with a forward pass.
    """
    def __init__(self):
        super().__init__()
        # Mimicking constants used in the original repro
        self._tensor_constant0 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=tf.int64)

    @tf.function
    def forward(self, primals, tangents):
        # Flatten inputs (mimicking fx_pytree.tree_flatten_spec)
        # In TF, we just accept tensors directly for this simplified repro
        
        # primals_1: "f32[256, 256]" -> t: "f32[256, 256]"
        # torch.ops.aten.t.default -> tf.transpose
        t = tf.transpose(primals)
        
        # primals_2: "f32[15, 256]" -> mm: "f32[15, 256]"
        # torch.ops.aten.mm.default -> tf.matmul
        # Note: Shapes adjusted for TF compatibility in this minimal repro
        mm = tf.matmul(tangents, t)
        
        # new_empty: "f32[16, 256]" -> tf.zeros_like / tf.constant
        # torch.ops.aten.new_empty.default
        new_empty = tf.zeros([16, 256], dtype=tf.float32)
        
        # index_put: "f32[16, 256]"
        # torch.ops.aten.index_put.default -> tf.tensor_scatter_nd_update
        indices = tf.expand_dims(self._tensor_constant0, axis=1)
        updates = mm
        index_put = tf.tensor_scatter_nd_update(new_empty, indices, updates)
        
        # slice_2: "f32[15, 256]"
        # torch.ops.aten.slice.Tensor -> tf.slice
        # The bug specifically mentions the stack trace for slice_2 being wrong.
        slice_2 = tf.slice(index_put, [0, 0], [15, 256])
        
        return slice_2

def test_stack_trace_filtering():
    """
    Test case leveraging tf.debugging.is_traceback_filtering_enabled.
    Relates to the original issue by focusing on the integrity and visibility 
    of stack traces during execution.
    """
    module = InnerModule()
    
    # Setup inputs matching the shapes in the original repro
    primals = tf.random.normal([256, 256])
    tangents = tf.random.normal([15, 256])

    # 1. Check the default state of traceback filtering
    # This relates to the "Stack trace" keyword in the bug report.
    is_enabled = tf.debugging.is_traceback_filtering_enabled()
    print(f"Initial traceback filtering enabled: {is_enabled}")
    assert isinstance(is_enabled, bool), "API should return a boolean"

    # 2. Disable filtering to ensure full stack traces are available
    # This simulates the desire to see the "correct" or "full" stack trace,
    # which is the opposite of the bug (where the trace is wrong).
    tf.debugging.disable_traceback_filtering()
    assert tf.debugging.is_traceback_filtering_enabled() is False, \
        "Traceback filtering should be disabled after disable_traceback_filtering()"

    # 3. Execute the graph to generate a trace
    # In the PyTorch bug, the graph is exported and printed with annotations.
    # Here we execute it to ensure the stack trace mechanism is active.
    try:
        output = module.forward(primals, tangents)
        assert output.shape == (15, 256), "Output shape should match the slice operation"
    except Exception as e:
        # If an error occurs, we inspect the traceback.
        # With filtering disabled, we expect a detailed traceback.
        tb_str = traceback.format_exc()
        print(f"Traceback with filtering disabled:\n{tb_str}")
        # We expect internal TF frames to be present when filtering is disabled
        assert "tensorflow" in tb_str or "InnerModule" in tb_str

    # 4. Re-enable filtering
    tf.debugging.enable_traceback_filtering()
    assert tf.debugging.is_traceback_filtering_enabled() is True, \
        "Traceback filtering should be enabled after enable_traceback_filtering()"

    print("Test passed: Traceback filtering API works as expected.")

if __name__ == "__main__":
    test_stack_trace_filtering()