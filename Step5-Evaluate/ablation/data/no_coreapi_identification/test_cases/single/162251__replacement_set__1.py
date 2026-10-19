import torch
import tensorflow as tf
import numpy as np
from functools import partial

# Replacement: torch.__version__ -> tf.nn.with_space_to_batch
print(tf.nn.with_space_to_batch)

# Replacement: torch.zeros -> torch.ones
# Keeping torch.complex32 to test the specific bug scenario
t = torch.ones((5, 5, 9, 3), dtype=torch.complex32)

input = [
    [545460846592],
    {},
    [t],
    {}
]

# Replacement: torch.nn.PixelShuffle -> torch.nn.PixelUnshuffle
# This tests the PyTorch side with the huge factor.
try:
    r1 = torch.nn.PixelUnshuffle(*input[0], **input[1])
    r2 = r1(*input[2], **input[3])
    print("PyTorch PixelUnshuffle executed.")
except Exception as e:
    print(f"PyTorch PixelUnshuffle failed: {e}")

# Replacement: torch.nn.PixelShuffle -> tf.nn.with_space_to_batch
# This tests the TensorFlow side.
# We need a TF tensor. The original logic uses `t`.
# We convert `t` to a TF tensor to pass it to the TF function.
# Note: TF doesn't support complex32 (float16 complex) directly in all ops, usually complex64.
# We'll try to convert.
try:
    # Convert torch tensor to numpy then to tensorflow
    # t is complex32 (float16 real/imag). TF complex64 is float32 real/imag.
    # We might need to cast.
    np_array = t.numpy().astype(np.complex64)
    tf_t = tf.constant(np_array)

    # with_space_to_batch signature: (input, dilation_rate, padding, op, ...)
    # The original logic passes a huge number (545460846592) as the first arg to PixelShuffle (upscale_factor).
    # This maps to dilation_rate in with_space_to_batch.
    # We need padding and op.
    # Padding: let's use 0 or valid.
    # Op: let's use identity or something simple.
    # The original bug is about the huge number causing FPE.
    # So we pass the huge number as dilation_rate.

    # The original code structure:
    # r1 = PixelShuffle(factor)
    # r2 = r1(tensor)
    # with_space_to_batch is a function, not a class.
    # So we just call it.
    # r2 = tf.nn.with_space_to_batch(tf_t, dilation_rate=545460846592, padding=[[0,0], [0,0], [0,0], [0,0]], op=lambda x: x)
    # Note: padding shape depends on input rank. Input is (5, 5, 9, 3). Rank 4.
    # Padding for spatial dims (usually last 2 or middle 2 depending on data_format).
    # Default data_format is NHWC. Spatial dims are H, W.
    # So padding is [[0,0], [0,0], [0,0], [0,0]] (Batch, H, W, Channels).
    # Wait, with_space_to_batch takes `padding` argument.
    # Let's try to call it.

    # However, the prompt asks to "replace ... in one single test case".
    # It doesn't explicitly ask to fix the TF call to be semantically perfect, but to "translate semantics appropriately".
    # The semantic of the bug is "Pass huge number to reshape op".
    # So passing huge number to dilation_rate is the translation.

    # But wait, the prompt says "replace torch.nn.PixelShuffle with tf.nn.with_space_to_batch".
    # In the original code: `r1 = torch.nn.PixelShuffle(*input[0],**input[1])`
    # `input[0]` is `[545460846592]`.
    # So `r1 = tf.nn.with_space_to_batch(545460846592)`?
    # No, `with_space_to_batch` needs the input tensor.
    # The original code separates init and call.
    # `r2 = r1(*input[2],**input[3])` -> `r2 = r1(t)`.
    # Since `with_space_to_batch` is a function, I can't really do `r1 = func; r2 = r1(t)`.
    # Unless I use `functools.partial`.
    # `r1 = functools.partial(tf.nn.with_space_to_batch, dilation_rate=545460846592, padding=..., op=...)`
    # `r2 = r1(tf_t)`
    # This preserves the structure `r1 = ...; r2 = r1(...)`.

    # Let's try this approach.

    # Define padding (zeros for 4D tensor NHWC)
    # Input shape (5, 5, 9, 3) -> Batch, H, W, C
    padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
    # Define op (identity)
    def op_fn(x):
        return x

    # r1 = tf.nn.with_space_to_batch(...)
    # We need to bind the huge number.
    # The original code unpacks input[0] which is the list containing the huge number.
    # `*input[0]` -> `545460846592`.
    # `with_space_to_batch` takes `dilation_rate`.
    # So we bind dilation_rate.

    # Note: `with_space_to_batch` signature is (input, dilation_rate, padding, op, ...)
    # We want to bind dilation_rate, padding, op.
    r1 = partial(tf.nn.with_space_to_batch, dilation_rate=545460846592, padding=padding, op=op_fn)

    # r2 = r1(*input[2], **input[3])
    # input[2] is [t]. But t is torch tensor.
    # We need to pass the TF tensor.
    # The prompt says "replace source APIs".
    # It doesn't say "replace torch tensor with tf tensor".
    # But if I pass a torch tensor to a TF function, it will error.
    # I will assume I should convert `t` to `tf_t` for this specific call, or just pass `t` and let it error (which is a valid test result).
    # However, to be "helpful", I should probably convert it.
    # But the prompt says "Keep the original test case logic".
    # The logic is `r2 = r1(t)`.
    # If I change `t` to `tf_t`, I am changing the logic (variable usage).
    # I will stick to `t` and see what happens, or convert it inside the call?
    # Actually, `t` is defined as `torch.ones`.
    # If I pass `t` to `tf.nn...`, it will raise TypeError.
    # This is a valid test case (testing that TF doesn't accept Torch tensors).
    # But the goal is to test the "Similar API" for the "Bug" (FPE).
    # To test the FPE, I need the TF function to run.
    # So I must convert.

    # Let's look at the "torch.complex32" -> "tf.nn.with_space_to_batch" replacement again.
    # Maybe this implies that the *input* to the function should be the result of `with_space_to_batch`? No.
    # I will ignore the dtype replacement as it's nonsensical and stick to `torch.complex32` for the PyTorch tensor.

    # Let's just create the TF tensor and use it.
    # I'll add a comment explaining the conversion for the sake of the test.

    # Actually, looking at the prompt "replace ... in one single test case",
    # maybe I should just replace the *lines*.
    # Line 1: `r1 = torch.nn.PixelShuffle(*input[0],**input[1])`
    # Replace with: `r1 = torch.nn.PixelUnshuffle(*input[0],**input[1])`
    # AND
    # Replace with: `r1 = tf.nn.with_space_to_batch(...)`
    # I can't do both on the same line.
    # I will create two separate blocks.

    # Block 1: PyTorch
    r1_pt = torch.nn.PixelUnshuffle(*input[0], **input[1])
    r2_pt = r1_pt(*input[2], **input[3])

    # Block 2: TensorFlow
    # To make it runnable, I need a TF tensor.
    # I'll create one from `t`.
    tf_t = tf.constant(t.numpy().astype(np.complex64))
    # To match the structure `r1 = ...; r2 = r1(...)`:
    r1_tf = partial(tf.nn.with_space_to_batch, dilation_rate=545460846592, padding=padding, op=op_fn)
    r2_tf = r1_tf(tf_t)

except Exception as e:
    print(f"TensorFlow with_space_to_batch error: {e}")