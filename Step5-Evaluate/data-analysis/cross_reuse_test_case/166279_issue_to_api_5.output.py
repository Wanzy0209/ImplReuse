import torch

def test_chunk_dynamo_divergence():
    """
    Test case for Issue 166279: [Fuzzer][Eager/Compile Divergence] 
    assert len(input_size) == len(new_size)
    
    This test leverages torch.backends.cuda.is_built to ensure the test
    runs in the appropriate environment (CUDA if available) as indicated
    by the bug report's context.
    """
    
    # Leverage the similar API to determine execution device
    # The bug report comments indicate 'device=cuda', so we check if CUDA is built.
    if torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not built, running test on CPU. Bug might be CUDA-specific.")

    torch._dynamo.config.capture_scalar_outputs = True
    torch.manual_seed(1166094474)

    def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
        # Reproduce the logic from the bug report
        var_node_3 = torch.full((12,), False, dtype=torch.bool, device=device)
        var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
        
        var_node_6 = arg_0
        var_node_7 = arg_1
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

    # Sentinel tensor to ensure gradient computation
    sentinel = torch.tensor(1.0, requires_grad=True, device=device)

    # Initialize arguments on the determined device
    # Using as_strided as per the original report to ensure specific memory layouts
    arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool().to(device), (12,), (1,))
    arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64).to(device), (10,), (1,))
    arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool().to(device), (6, 4), (4, 1))
    arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool().to(device), (2,), (1,))

    args = (arg_0, arg_1, arg_2, arg_3, sentinel)

    # Run in Eager mode
    try:
        result_original = fuzzed_program(*args)
        print(' eager success')
    except Exception as e:
        print(f' eager failed: {e}')
        raise

    # Run in Compiled mode
    try:
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' compile success')
    except Exception as e:
        print(f' compile failed: {e}')
        raise

    # Check for divergence
    assert torch.equal(result_original, result_compiled), \
        f"Divergence detected between eager and compiled modes.\nEager: {result_original}\nCompiled: {result_compiled}"

if __name__ == "__main__":
    test_chunk_dynamo_divergence()