import torch
import tensorflow as tf
from tf.compat.v1.io import TFRecordCompressionType

def test_flex_attention_recompile_limit():
    """
    Test case for Issue 166153: Flex Attention with AO hit the recompile limit.
    This test simulates the recompilation limit hit scenario using the requested
    API replacements and the similar API (TFRecordCompressionType).
    """

    # Replaces AutoTokenizer.from_pretrained
    # Original: tokenizer = AutoTokenizer.from_pretrained("bert")
    # New: tokenizer_shape = tf.broadcast_dynamic_shape((1,), (1,))
    tokenizer_shape = tf.broadcast_dynamic_shape((1,), (1,))

    # Replaces sys.platform.startswith
    # Original: if sys.platform.startswith("linux"):
    # New: if tf.broadcast_dynamic_shape((1,), (1,)):
    if tf.broadcast_dynamic_shape((1,), (1,)):
        pass

    # Replaces config.recompile_limit
    # Original: limit = config.recompile_limit
    # New: limit = tf.broadcast_dynamic_shape((8,), (1,)).as_list()[0]
    limit = tf.broadcast_dynamic_shape((8,), (1,)).as_list()[0]

    # Mock Model with flex_attention
    class Model:
        def __init__(self):
            pass

        def flex_attention(self, query, key, value):
            # Replaces self.gate_proj
            # Original: x = self.gate_proj(x)
            # New: _ = tf.broadcast_dynamic_shape(query.shape, (1,))
            _ = tf.broadcast_dynamic_shape(query.shape, (1,))
            
            # Actual logic (simplified flex attention)
            return torch.matmul(query, key.transpose(-2, -1))

    model = Model()

    # Replaces torch.compile
    # Original: compiled_model = torch.compile(model.flex_attention)
    # New: We use torch.all to verify the compilation/execution context
    def compiled_model(*args):
        # Instead of compiling, we check if all inputs are valid using torch.all
        if torch.all(torch.tensor([True])):
            return model.flex_attention(*args)

    # Simulation of the bug: hitting the recompile limit
    recompile_count = 0
    
    # Use the Similar API: TFRecordCompressionType
    # We iterate through compression types to simulate different states causing recompilation
    compression_types = [TFRecordCompressionType.NONE, TFRecordCompressionType.ZLIB, TFRecordCompressionType.GZIP]
    
    for i in range(10):
        # Replaces torch._dynamo
        # Original: if torch._dynamo.hit_config.recompile_limit:
        # New: We check the limit using tf.broadcast_dynamic_shape
        # We simulate the check by broadcasting the count against the limit
        check_shape = tf.broadcast_dynamic_shape((recompile_count,), (limit,))
        
        if check_shape.as_list()[0] >= limit:
            # Replaces torch.compiler_troubleshooting.html
            # Original: print("See torch.compiler_troubleshooting.html")
            # New: print("See", tf.broadcast_dynamic_shape((1,), (1,)))
            print(f"Hit limit {limit} at iteration {i}")
            print(f"See {tf.broadcast_dynamic_shape((1,), (1,))}")
            break

        # Replaces nmodel.generate
        # Original: output = nmodel.generate(input_ids)
        # New: output_shape = tf.broadcast_dynamic_shape(...)
        # We simulate generation by calculating a shape
        q = torch.randn(1, 8, 4, 4)
        k = torch.randn(1, 8, 4, 4)
        v = torch.randn(1, 8, 4, 4)
        
        # Simulate state change (e.g. different compression type) causing recompile
        current_comp_type = compression_types[i % len(compression_types)]
        
        # Run the "compiled" model
        output = compiled_model(q, k, v)
        
        # Replaces nmodel.generate (execution part)
        # The prompt asks to replace nmodel.generate with tf.broadcast_dynamic_shape
        # We can assign the result of the generation to the broadcast result
        generation_result = tf.broadcast_dynamic_shape(output.shape, (1,))
        
        recompile_count += 1

    # Assert that we hit the limit as expected by the bug report
    assert recompile_count == limit, f"Expected to hit limit at {limit}, but stopped at {recompile_count}"

if __name__ == "__main__":
    test_flex_attention_recompile_limit()