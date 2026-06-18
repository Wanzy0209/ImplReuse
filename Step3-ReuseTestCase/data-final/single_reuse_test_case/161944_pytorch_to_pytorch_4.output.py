import torch

if torch.cuda.is_available():
    torch.set_default_device('cuda')

    inp = torch.randn(8192)
    func = torch.sigmoid
    out1 = func(inp)
    out2 = torch.compile(func)(inp)
    out3_high = func(inp.to(torch.float64))

    diff_eager = (out3_high - out1).abs().max()
    diff_compiled = (out3_high - out2).abs().max()

    print(f"Difference (Eager vs High): {diff_eager}")
    print(f"Difference (Compiled vs High): {diff_compiled}")

    # Check if the compiled result deviates significantly more than the eager result
    # compared to the high precision ground truth.
    # This assertion helps detect if "fast math" optimizations are degrading accuracy.
    assert diff_compiled < diff_eager * 10, "Compiled result has significantly higher error than eager result"
else:
    print("CUDA not available, skipping test.")