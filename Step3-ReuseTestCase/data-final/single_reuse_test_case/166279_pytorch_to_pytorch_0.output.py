import torch
import torch._dynamo

# Reproduce the issue from Bug Report 166279
# Issue: [Fuzzer][Eager/Compile Divergence] assert len(input_size) == len(new_size)
# API Under Test: torch.chunk

def test_chunk_compile_divergence():
    # Configuration from the bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch.manual_seed(1166094474)

    # Setup inputs based on the fuzzer output
    # 1D input: size=(12,), dtype=bool
    input_1d = torch.full((12,), False, dtype=torch.bool)
    
    # 2D input: size=(6, 4), stride=(4, 1), dtype=bool
    # Using as_strided to match the specific memory layout from the fuzzer
    input_2d = torch.as_strided(
        torch.randint(0, 2, (24,), dtype=torch.int8).bool(), 
        (6, 4), 
        (4, 1)
    )

    def func_to_test(x, y):
        # Operation 1: Chunk 1D tensor
        # Corresponds to: var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
        # Input size (12,), chunks 4 -> Output size (3,)
        res_1 = torch.chunk(x, 4, dim=0)[0]
        
        # Operation 2: Chunk 2D tensor along dim 1
        # Corresponds to: var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
        # Input size (6, 4), chunks 4 -> Output size (6, 1)
        res_2 = torch.chunk(y, 4, dim=1)[0]
        
        # Operation 3: Squeeze the result of the 2D chunk
        # Corresponds to: var_node_8 = torch.squeeze(var_node_9)
        # Input size (6, 1) -> Output size (6,)
        res_2_squeezed = torch.squeeze(res_2)
        
        # Operation 4: Concatenate the 1D results
        # Corresponds to: var_node_1 = torch.cat([var_node_2, var_node_8], dim=0)
        # Input sizes (3,) and (6,) -> Output size (9,)
        return torch.cat([res_1, res_2_squeezed], dim=0)

    # Test Eager execution
    try:
        eager_result = func_to_test(input_1d, input_2d)
        print(" Eager execution successful.")
    except Exception as e:
        print(f" Eager execution failed: {e}")
        return

    # Test Compiled execution
    try:
        compiled_func = torch.compile(func_to_test, fullgraph=True, dynamic=True)
        compiled_result = compiled_func(input_1d, input_2d)
        
        # Verify results match
        if torch.equal(eager_result, compiled_result):
            print(" Compiled execution successful and matches eager.")
        else:
            print(" Divergence: Eager and Compiled results differ.")
    except AssertionError as e:
        # Catching the specific assertion error mentioned in the bug report title
        print(f" Compilation failed with AssertionError: {e}")
    except Exception as e:
        print(f" Compilation failed with Exception: {e}")

if __name__ == "__main__":
    test_chunk_compile_divergence()