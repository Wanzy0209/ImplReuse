import torch

def test_squeeze_eager_compile_divergence():
    """
    Test case for Issue 166270.
    Verifies that torch.squeeze behaves consistently between eager and compiled modes
    when reducing a 1D tensor to a 0D tensor and subsequently reshaping.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch.manual_seed(1061983224)

    # Setup input tensor
    # Using as_strided to mimic the fuzzer's input generation
    arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))
    sentinel = torch.tensor(1.0, requires_grad=True)

    def program(x, sentinel):
        # The specific sequence of operations triggering the divergence
        var_1 = torch.chunk(x, 4, dim=0)[0]  # size (1,)
        var_2 = torch.squeeze(var_1)        # size () (Scalar)
        var_3 = torch.stack([var_2], dim=0) # size (1,)
        var_4 = torch.reshape(var_3, [1])   # size (1,)
        return var_4 * sentinel

    # 1. Run in Eager mode
    result_eager = program(arg_0, sentinel)

    # 2. Run in Compiled mode (fullgraph=True, dynamic=True triggers the bug)
    compiled_program = torch.compile(program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(arg_0, sentinel)

    # 3. Assert consistency
    # If the bug exists, the compiled run will raise an assertion error internally.
    # If fixed, results should match.
    assert torch.equal(result_eager, result_compiled), \
        f"Eager and compiled results diverge.\nEager: {result_eager}\nCompiled: {result_compiled}"

if __name__ == "__main__":
    test_squeeze_eager_compile_divergence()
    print("Test passed successfully.")