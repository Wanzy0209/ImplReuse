import torch
import unittest

class TestChunkCompileDivergence(unittest.TestCase):
    def setUp(self):
        # Configuration from the bug report that triggers the divergence
        torch._dynamo.config.capture_scalar_outputs = True
        torch.manual_seed(1166094474)

    def test_chunk_gather_squeeze_cat_divergence(self):
        """
        Reproduces Issue 166279: Eager/Compile Divergence with torch.chunk.
        The bug involves an assertion failure: assert len(input_size) == len(new_size)
        when running torch.compile on a specific sequence of chunk, gather, squeeze, and cat ops.
        """
        
        # Sentinel tensor to ensure gradient computation
        sentinel = torch.tensor(1.0, requires_grad=True)

        # Create inputs with specific strides using as_strided, which is crucial for the bug
        arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
        arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
        arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
        arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))

        def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
            # Logic extracted from the fuzzer report
            var_node_3 = torch.full((12,), False, dtype=torch.bool)
            var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
            
            var_node_6 = arg_0
            _input_size_var_node_5 = var_node_6.size(0)
            _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
            var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
            var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
            
            var_node_10 = arg_2
            var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
            var_node_8 = torch.squeeze(var_node_9)
            
            var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
            var_node_11 = arg_3
            var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
            
            # Ensure gradient computation
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        args = (arg_0, arg_1, arg_2, arg_3, sentinel)

        # 1. Run in Eager mode
        try:
            result_eager = fuzzed_program(*args)
            print(" Eager execution succeeded")
        except Exception as e:
            self.fail(f"Eager mode failed unexpectedly: {e}")

        # 2. Run in Compiled mode (fullgraph=True, dynamic=True as per report)
        try:
            compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
            result_compiled = compiled_program(*args)
            print(" Compile execution succeeded")
        except AssertionError as e:
            # This is the specific error mentioned in the bug report title
            if "len(input_size) == len(new_size)" in str(e):
                self.fail(f"Bug reproduced: Compile mode failed with assertion error: {e}")
            else:
                raise
        except Exception as e:
            self.fail(f"Compile mode failed with unexpected error: {e}")

        # 3. Verify results match (if compilation succeeded)
        # Note: If the bug is present, this line might not be reached.
        self.assertTrue(torch.allclose(result_eager, result_compiled), 
                        "Divergence detected between eager and compiled results")

if __name__ == "__main__":
    unittest.main()