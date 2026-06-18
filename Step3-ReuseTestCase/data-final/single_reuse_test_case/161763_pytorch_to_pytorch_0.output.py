import torch

def test_compile_neg_add_uint():
    """
    Test case for Issue 161763: neg+add computation including uint tensor is incorrect under inductor.
    Verifies that torch.compile handles the combination of torch.neg on uint8 tensors
    followed by addition with float32 tensors correctly.
    """
    def foo(x):
        c = torch.tensor(7, dtype=torch.uint8)
        return c + x, torch.neg(c), torch.neg(c) + x

    torch.manual_seed(0)
    x = torch.randn(2, 2, dtype=torch.float32)

    # Run eager mode
    res = foo(x)

    # Run compiled mode (default backend is inductor)
    cfoo = torch.compile(foo)
    cres = cfoo(x)

    # Check the specific failing case: torch.neg(c) + x
    # In eager mode, neg(uint8(7)) wraps to 249. 249 + x results in values > 200.
    # In the buggy compiled version, neg(uint8(7)) was treated as -7, resulting in negative values.
    assert torch.allclose(res[0], cres[0]), "Mismatch in c + x"
    assert torch.equal(res[1], cres[1]), "Mismatch in torch.neg(c)"
    assert torch.allclose(res[2], cres[2]), "Mismatch in torch.neg(c) + x (Bug 161763)"

if __name__ == "__main__":
    test_compile_neg_add_uint()