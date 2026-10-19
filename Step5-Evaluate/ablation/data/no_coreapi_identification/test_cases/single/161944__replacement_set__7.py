import torch
import torch.nn.functional as F
import tensorflow as tf

# Use the similar API to check environment
if not tf.test.is_built_with_rocm():
    print("Test skipped: ROCm not supported")
else:
    # Replacement: torch.set_default_device('cuda') -> tf.keras.activations.serialize('cuda')
    tf.keras.activations.serialize('cuda')

    # Replacement: torch.randn(8192) -> torch.nn.functional.max_pool2d
    # We need an input for max_pool2d. Using torch.ones to avoid using torch.randn.
    dummy_input = torch.ones(1, 1, 128, 64) # 8192 elements
    inp = F.max_pool2d(dummy_input, kernel_size=2)

    # Replacement: torch.exp -> tf.keras.activations.serialize
    func = tf.keras.activations.serialize

    # Original: out1 = func(inp)
    out1 = func(inp)

    # Replacement: torch.compile -> torch.lobpcg
    # Adapting the call structure to be runnable with the new API
    # torch.lobpcg returns (eigenvalues, eigenvectors)
    out2 = torch.lobpcg(inp, k=1)[0]

    # Replacement: inp.to(torch.float64) -> tf.keras.activations.serialize(inp)
    # Replacement: torch.float64 -> tf.keras.activations.serialize
    # Combining these, we serialize the input.
    out3_high = func(tf.keras.activations.serialize(inp))

    # Original logic: print differences
    # Note: out1 and out3_high are strings, out2 is a tensor.
    # This will raise a TypeError, but we preserve the logic structure.
    try:
        print((out3_high - out1).abs().max())
        print((out3_high - out2).abs().max())
    except Exception as e:
        print(f"Error during comparison: {e}")