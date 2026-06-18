import torch
import unittest

class TestCholeskySimilarAPI(unittest.TestCase):
    def test_cholesky_dynamo_divergence(self):
        """
        Test case adapted from a fuzzer report for torch.gather (Issue 166245).
        This test targets the similar API torch.cholesky to verify behavior
        under torch._dynamo compilation.
        """
        # Configuration from the original bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch.manual_seed(751735337)

        # The original bug was triggered on CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        device = torch.device("cuda")

        def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
            # --- Replicated logic from the original fuzzer ---
            var_node_4 = arg_0 # size=(15, 108, 4), stride=(432, 1, 4), dtype=int16, device=cuda
            var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0] # size=(15, 27, 4), stride=(108, 4, 1), dtype=int16, device=cuda
            var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] # size=(15, 27, 1), stride=(27, 1, 1), dtype=int16, device=cuda
            var_node_1 = torch.squeeze(var_node_2) # size=(15, 27), stride=(1, 1), dtype=int16, device=cuda
            
            var_node_8 = torch.full((13, 27), 3, dtype=torch.int16, device=device) # size=(13, 27), stride=(27, 1), dtype=int16, device=cuda
            var_node_9 = arg_1 # size=(11,), stride=(1,), dtype=int64, device=cuda
            _input_size_var_node_7 = var_node_8.size(0)
            _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), device=var_node_8.device)
            var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7) # size=(11, 27), stride=(1, 1), dtype=int16, device=cuda
            var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0) # size=(11, 27), stride=(1, 1), dtype=int16, device=cuda
            
            var_node_12 = arg_2 # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
            var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0) # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
            
            var_node_13 = torch.full((1,), 3, dtype=torch.int64, device=device) # size=(1,), stride=(1,), dtype=int64, device=cuda
            _input_size_var_node_10 = var_node_11.size(0)
            _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), device=var_node_11.device)
            var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10) # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
            
            var_node_16 = arg_3 # size=(1, 27), stride=(1, 1), dtype=int16, device=cuda
            var_node_17 = arg_4 # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
            var_node_18 = arg_5 # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
            var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0) # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
            
            var_node_20 = arg_6 # size=(1,), stride=(1,), dtype=int64, device=cuda
            var_node_19 = torch.clamp(var_node_20, min=None, max=1.0) # size=(1,), stride=(1,), dtype=int64, device=cuda
            _input_size_var_node_14 = var_node_15.size(0)
            _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), device=var_node_15.device)
            var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14) # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
            
            # --- Adaptation for torch.cholesky ---
            # Original code created var_node_22 with shape (4, 27).
            # torch.cholesky requires a square matrix (N, N) and floating point dtype.
            # We adapt the creation to produce a (4, 4) positive-definite matrix.
            
            # Create a base matrix
            var_node_22 = torch.full((4, 4), 1.0, dtype=torch.float32, device=device)
            # Add a diagonal offset to ensure positive definiteness
            var_node_22 = var_node_22 + 3.0 * torch.eye(4, device=device)
            
            # Call the similar API: torch.cholesky
            # Replaces the original torch.gather call site
            result = torch.cholesky(var_node_22)
            
            return result

        # Initialize inputs matching the original fuzzer shapes and dtypes
        arg_0 = torch.randn(15, 108, 4, dtype=torch.int16, device=device)
        arg_1 = torch.randint(0, 100, (11,), dtype=torch.int64, device=device)
        arg_2 = torch.randn(3, 27, dtype=torch.int16, device=device)
        arg_3 = torch.randn(1, 27, dtype=torch.int16, device=device)
        arg_4 = torch.randn(1, 27, dtype=torch.int16, device=device)
        arg_5 = torch.randn(1, 27, dtype=torch.int16, device=device)
        arg_6 = torch.randint(0, 100, (1,), dtype=torch.int64, device=device)
        sentinel = None

        # 1. Run Eager mode
        try:
            eager_output = fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel)
        except Exception as e:
            self.fail(f"Eager execution failed with: {e}")

        # 2. Run Compiled mode (torch._dynamo)
        compiled_fn = torch.compile(fuzzed_program)
        try:
            compiled_output = compiled_fn(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel)
        except Exception as e:
            # The original bug was an assertion failure in dynamo.
            # If this occurs, the test reproduces the issue.
            self.fail(f"Compiled execution failed with: {e}")

        # 3. Verify outputs match
        self.assertTrue(torch.allclose(eager_output, compiled_output))

if __name__ == "__main__":
    unittest.main()