import torch
import pytest

# Reproduction of Issue 165081: [Fuzzer][Eager/Compile Divergence]
# Could not guard on data-dependent expression Ne(u0, 9)
#
# The issue involves a complex graph of matrix multiplications leading to a 
# data-dependent shape operation (likely torch.nonzero based on the API metadata)
# that causes torch._dynamo to fail guarding.

def test_issue_165081_eager_compile_divergence():
    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch.manual_seed(52676)

    # Determine device (CUDA is preferred as per original report, fallback to CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize arguments based on the usage in the fuzzed_program
    # Shapes and dtypes are inferred from the comments in the original code
    arg_0 = torch.randn((9, 9, 9), dtype=torch.float64, device=device)
    arg_1 = torch.randn((9, 9, 11), dtype=torch.float64, device=device)
    arg_2 = torch.randn((9, 12, 8), dtype=torch.float64, device=device)
    arg_3 = torch.randn((9, 8, 13), dtype=torch.float64, device=device)
    arg_4 = torch.randn((9, 13, 7), dtype=torch.float64, device=device)
    arg_5 = torch.randn((9, 7, 16), dtype=torch.float64, device=device)
    arg_6 = torch.randn((9, 16, 12), dtype=torch.float64, device=device)
    arg_7 = torch.randn((9, 12, 11), dtype=torch.float64, device=device)
    
    # Remaining arguments are unused in the visible snippet but required by signature
    dummy_shape = (1,)
    arg_8 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_9 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_10 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_11 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_12 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_13 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_14 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_15 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_16 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_17 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    arg_18 = torch.randn(dummy_shape, dtype=torch.float64, device=device)
    sentinel = None

    # The fuzzed_program logic
    # Note: The original code was truncated. Based on the "Original API Under Test: torch.nonzero"
    # and the error "Could not guard on data-dependent expression", we append torch.nonzero
    # to the end of the provided graph to trigger the divergence.
    def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
        var_node_6 = arg_0
        var_node_7 = arg_1
        var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
        var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64, device=device)
        var_node_10 = arg_2
        var_node_8 = torch.matmul(var_node_9.to(torch.float64), var_node_10.to(torch.float64))
        var_node_4 = torch.matmul(var_node_5.to(torch.float64), var_node_8.to(torch.float64))
        var_node_13 = arg_3
        var_node_14 = arg_4
        var_node_12 = torch.matmul(var_node_13.to(torch.float64), var_node_14.to(torch.float64))
        var_node_15 = arg_5
        var_node_11 = torch.matmul(var_node_12.to(torch.float64), var_node_15.to(torch.float64))
        var_node_3 = torch.matmul(var_node_4.to(torch.float64), var_node_11.to(torch.float64))
        var_node_17 = arg_6
        var_node_18 = arg_7
        var_node_16 = torch.matmul(var_node_17.to(torch.float64), var_node_18.to(torch.float64))
        var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_16.to(torch.float64))
        var_node_23 = torch.full((156, 8), -0.5249394453404403, dtype=torch.float64, device=device)
        var_node_24 = torch.full((8, 9), 0.9331226188585692, dtype=torch.float64, device=device)
        var_node_22 = torch.matmul(var_node_23.to(torch.float64), var_node_24.to(torch.float64))
        var_node_26 = torch.full((9, 13), -0.9276381954691514, dtype=torch.float64, device=device)
        
        # Leveraging the API under test (torch.nonzero) which likely triggers the guard failure
        # due to data-dependent output shapes.
        return torch.nonzero(var_node_26)

    # Compile the function
    # The error "Could not guard on data-dependent expression Ne(u0, 9)" occurs during compilation/execution
    try:
        compiled_fn = torch.compile(fuzzed_program)
        result = compiled_fn(
            arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 
            arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, 
            arg_16, arg_17, arg_18, sentinel
        )
        # If the bug is fixed, this should execute without error.
        # We can assert the result is a tensor (nonzero returns indices).
        assert isinstance(result, torch.Tensor)
        assert result.device == device
    except Exception as e:
        # If the bug exists, it might raise an error related to guards or divergence.
        # For the purpose of a test case verifying a fix, we might want to let the exception propagate
        # or check for the specific error message. Here we print it for visibility.
        print(f"Exception during execution: {e}")
        raise

if __name__ == "__main__":
    test_issue_165081_eager_compile_divergence()