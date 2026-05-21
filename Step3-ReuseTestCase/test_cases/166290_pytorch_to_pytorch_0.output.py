import torch
import pytest

def test_squeeze_compile_divergence():
    """
    Regression test for Issue 166290.
    Verifies that torch.squeeze works correctly with torch.compile
    on tensors resulting from index_select and chunk operations.
    """
    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch.manual_seed(974450504)

    # Use CUDA if available, otherwise CPU (original bug was on CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Recreate arg_0: size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool
    # This specific strided setup is crucial for triggering the bug
    arg_0 = torch.as_strided(
        torch.randint(0, 2, (26010,), dtype=torch.int8, device=device).bool(),
        (17, 30, 17, 3),
        (1530, 51, 3, 1)
    )

    def fuzzed_program(input_tensor):
        # Operations leading to the squeeze call
        # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
        chunked = torch.chunk(input_tensor, 3, dim=3)[0]
        
        # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
        # Generate indices dynamically to match the original behavior
        input_size = chunked.size(0)
        indices = torch.randint(0, input_size, (15,), device=chunked.device)
        indexed = torch.index_select(chunked, 0, indices)
        
        # The API under test: torch.squeeze
        # var_node_0 = torch.squeeze(var_node_1)
        return torch.squeeze(indexed)

    # 1. Run in Eager mode
    try:
        result_eager = fuzzed_program(arg_0)
    except Exception as e:
        pytest.fail(f"Eager mode failed with: {e}")

    # 2. Run in Compiled mode
    # The bug manifests as an IndexError during compilation or execution
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    try:
        result_compiled = compiled_program(arg_0)
    except Exception as e:
        pytest.fail(f"Compiled mode failed with: {e}")

    # 3. Verify results match
    assert torch.equal(result_eager, result_compiled), \
        "Divergence between eager and compiled results"

if __name__ == "__main__":
    test_squeeze_compile_divergence()
    print(" Test passed")