import torch
import tensorflow as tf

# Replacement: torch.device -> torch.diff
# We create a dummy tensor to apply torch.diff to satisfy the replacement.
device = torch.diff(torch.tensor([1.0, 2.0]))

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960

    # Replacement: torch.randn -> torch.var_mean
    # We use torch.var_mean to generate the input data. 
    # Since var_mean reduces dimensions, we expand the result to match the required shape.
    # Note: torch.bfloat16 is replaced by tf.nn.with_space_to_batch, which cannot be used as a dtype argument.
    # We proceed with the default dtype of var_mean (float32).
    rand_data = torch.randn(N, C, H, W)
    var_val, _ = torch.var_mean(rand_data)
    x = var_val.expand(N, C, H, W)

    # Replacement: x.to -> tf.nn.with_space_to_batch
    # Replacement: torch.channels_last -> tf.nn.with_space_to_batch
    # We convert the torch tensor to a tensorflow tensor to use the replacement API.
    # We also transpose to NHWC (channels_last) as implied by the context.
    x_tf = tf.constant(x.numpy())
    x_tf = tf.transpose(x_tf, [0, 2, 3, 1])
    
    # Apply the replacement for x.to(memory_format=...)
    x_tf = tf.nn.with_space_to_batch(
        input=x_tf,
        dilation_rate=[1, 1],
        padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
        op=lambda t: t
    )

    # Replacement: x.stride -> tf.nn.with_space_to_batch
    # Original code printed x.stride(). We replace the call with the API.
    stride_info = tf.nn.with_space_to_batch(
        input=x_tf,
        dilation_rate=[1, 1],
        padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
        op=lambda t: t
    )
    print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Input stride: {stride_info.shape}")

    # Replacement: nn.MaxPool2d -> tf.nn.with_space_to_batch
    # We define the pooling operation to be wrapped.
    # Original: kernel_size=3, stride=2, padding=1
    def pool_op(input_tensor):
        return tf.nn.max_pool2d(input_tensor, ksize=3, strides=2, padding='VALID')

    # Apply with_space_to_batch to perform the pooling.
    # We adjust padding to match the original behavior (padding=1).
    y = tf.nn.with_space_to_batch(
        input=x_tf,
        dilation_rate=[1, 1],
        padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
        op=pool_op
    )

    # Convert back to torch for the checks, as the check replacements are torch functions.
    y_torch = torch.from_numpy(y.numpy())

    # Replacement: torch.isnan -> torch.isinf
    # Original: torch.isnan(y).any()
    has_nan = torch.isinf(y_torch).any().item()

    # Replacement: torch.isinf -> torch.isnan
    # Original: torch.isinf(y).any()
    has_inf = torch.isnan(y_torch).any().item()

    # Replacement: y.min -> tf.nn.with_space_to_batch
    # Original: y.min()
    # We use the replacement API to get the "min" value.
    min_res = tf.nn.with_space_to_batch(
        input=y,
        dilation_rate=[1, 1],
        padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
        op=lambda t: t
    )
    # Extract a scalar for the print statement
    min_val = min_res.numpy().flatten()[0]

    print(f"Output contains NaN? {has_nan}")
    print(f"Output contains Inf? {has_inf}")
    print(f"Stats: min={min_val}, max=...")

    if has_nan:
        print("Detected NaNs in MaxPool output!")

test_maxpool_nan()