import torch

def test_nonzero_compile():
    # Setup input tensor on CUDA to match the original bug environment
    # Create a tensor with specific zero/non-zero patterns
    x = torch.tensor([[0.0, 1.0, 0.0], 
                      [2.0, 0.0, 3.0]], device="cuda")

    # Define function using the similar API (torch.nonzero)
    # We specifically test the 'as_tuple' parameter to ensure it is not ignored
    # during compilation, similar to how alpha/beta were ignored in the addmm bug.
    f = lambda x: torch.nonzero(x, as_tuple=True)

    # Compile the function
    fc = torch.compile(f)

    # Run eager and compiled versions
    res_eager = f(x)
    res_compiled = fc(x)

    # Verify that the 'as_tuple' parameter was respected in the compiled version
    # by checking the type of the output.
    assert isinstance(res_eager, tuple), "Eager mode output should be a tuple"
    assert isinstance(res_compiled, tuple), "Compiled mode output should be a tuple (parameter ignored?)"

    # Verify the contents of the tensors match
    assert len(res_eager) == len(res_compiled), "Tuple length mismatch"
    for eager_tensor, compiled_tensor in zip(res_eager, res_compiled):
        assert torch.equal(eager_tensor, compiled_tensor), "Tensor content mismatch between eager and compiled"

    print("Test passed: torch.nonzero respects parameters in torch.compile")

if __name__ == "__main__":
    test_nonzero_compile()