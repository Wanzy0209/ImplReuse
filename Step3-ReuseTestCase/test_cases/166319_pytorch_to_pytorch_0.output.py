import torch

def test_compile_divergence_issue_166319():
    """
    Test case for Issue 166319: [Fuzzer][Eager/Compile Divergence] LoweringException assert bool(static_expr)
    
    This test verifies that torch.compile produces the same result as eager execution
    for a specific graph involving squeeze, chunk, gather, and specific strides.
    """
    # Check for CUDA availability as the bug report specifies device=cuda
    if not torch.cuda.is_available():
        print("Test skipped: CUDA not available")
        return

    # Configuration from the bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch.manual_seed(1215252001)

    # The function under test
    def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
        var_node_4 = arg_0 # size=(9, 1, 15, 4), stride=(60, 60, 0, 1), dtype=int32, device=cuda
        var_node_3 = torch.squeeze(var_node_4) # size=(9, 15, 4), stride=(60, 4, 1), dtype=int32, device=cuda
        var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] # size=(9, 15, 1), stride=(1, 1, 1), dtype=int32, device=cuda
        var_node_1 = torch.squeeze(var_node_2) # size=(9, 15), stride=(0, 1), dtype=int32, device=cuda
        var_node_7 = arg_1 # size=(20, 15), stride=(15, 1), dtype=int64, device=cuda
        var_node_8 = arg_2 # size=(18, 15), stride=(15, 1), dtype=int64, device=cuda
        _input_size_var_node_6 = var_node_7.size(0)
        _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
        var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6) # size=(18, 15), stride=(0, 0), dtype=int64, device=cuda
        var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0] # size=(9, 15), stride=(0, 1), dtype=int64, device=cuda
        var_node_0 = torch.mul(var_node_1, var_node_5) # size=(9, 15), stride=(0, 1), dtype=int64, device=cuda
        # Ensure gradient computation by multiplying with sentinel and taking real part
        result = var_node_0 * sentinel
        if result.is_complex():
            result = result.real
        return result

    # Input generation
    # Note: Explicitly moving to CUDA to match the bug report context
    sentinel = torch.tensor(1.0, requires_grad=True).cuda()
    arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1)).cuda()
    arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1)).cuda()
    arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1)).cuda()

    args = (arg_0, arg_1, arg_2, sentinel)

    # 1. Run Eager
    try:
        result_eager = fuzzed_program(*args)
        print(" Eager execution successful")
    except Exception as e:
        print(f" Eager execution failed: {e}")
        return

    # 2. Run Compiled (torch.compile)
    # Using the parameters from the bug report: fullgraph=True, dynamic=True
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    try:
        result_compiled = compiled_program(*args)
        print(" Compiled execution successful")
    except Exception as e:
        print(f" Compiled execution failed (Bug reproduced): {e}")
        raise

    # 3. Verify Consistency
    # The bug title mentions "Eager/Compile Divergence", so we check if results match
    assert torch.equal(result_eager, result_compiled), "Divergence detected between eager and compiled outputs"
    print(" Test passed: Eager and Compiled outputs match")

if __name__ == "__main__":
    test_compile_divergence_issue_166319()