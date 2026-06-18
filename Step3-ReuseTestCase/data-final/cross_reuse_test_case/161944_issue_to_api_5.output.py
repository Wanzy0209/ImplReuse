import torch
from torch.utils.cpp_extension import verify_ninja_availability

def test_exp_compile_precision():
    """
    Test case to reproduce the precision issue with torch.compile and torch.exp.
    Leverages verify_ninja_availability to ensure the build environment is ready.
    """
    # Check for CUDA availability as per the original bug report
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Leverage the similar API to ensure build dependencies are met.
    # torch.compile (Inductor) may require Ninja for C++ code generation,
    # so we verify availability before attempting compilation.
    try:
        verify_ninja_availability()
    except RuntimeError as e:
        print(f"Build dependency check failed: {e}. Skipping test.")
        return

    torch.set_default_device('cuda')

    # Original bug reproduction logic
    inp = torch.randn(8192)
    func = torch.exp
    
    # Eager execution
    out1 = func(inp)
    
    # Compiled execution
    out2 = torch.compile(func)(inp)
    
    # High precision reference
    out3_high = func(inp.to(torch.float64))
    
    # Calculate differences
    diff_eager = (out3_high - out1).abs().max()
    diff_compiled = (out3_high - out2).abs().max()
    
    print(f"Max difference (Eager vs High Precision): {diff_eager}")
    print(f"Max difference (Compiled vs High Precision): {diff_compiled}")
    
    # Basic assertions to ensure execution completed
    assert out1.shape == inp.shape
    assert out2.shape == inp.shape
    assert out3_high.shape == inp.shape

if __name__ == "__main__":
    test_exp_compile_precision()