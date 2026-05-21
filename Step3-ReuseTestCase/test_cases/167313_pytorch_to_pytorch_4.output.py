import torch

def test_argmax_compile():
    # Setup input tensor on CUDA to match the original bug context
    x = torch.rand(2, 3, device="cuda")

    # Define a function using torch.argmax with specific parameters.
    # We test 'dim' and 'keepdim' to ensure they are not ignored by the compiler,
    # similar to how 'alpha' and 'beta' were ignored in the addmm bug.
    f = lambda x: torch.argmax(x, dim=1, keepdim=True)

    # Compile the function
    fc = torch.compile(f)

    # Execute in eager mode
    res_eager = f(x)
    
    # Execute in compiled mode
    res_compiled = fc(x)

    # Verify that the results are identical
    assert torch.equal(res_eager, res_compiled), f"Values mismatch:\nEager: {res_eager}\nCompiled: {res_compiled}"
    
    # Verify that the shape is preserved (specifically checking keepdim behavior)
    assert res_eager.shape == res_compiled.shape, f"Shape mismatch:\nEager: {res_eager.shape}\nCompiled: {res_compiled.shape}"

if __name__ == "__main__":
    test_argmax_compile()