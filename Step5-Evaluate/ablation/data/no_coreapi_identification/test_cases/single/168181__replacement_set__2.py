import torch
import triton
import triton.language as tl
import tensorflow as tf

# Define the Triton kernel used in the test
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load data
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    # Compute and store
    output = x + y
    tl.store(output_ptr + offsets, output)

def test_triton_kernel_replacements():
    # Set seed for reproducibility, specifically for LazyLinear weight initialization
    torch.manual_seed(0)
    
    device = torch.device("cuda")

    # Replacement: torch.randn -> tf.random.stateless_truncated_normal
    # We generate the inputs using TensorFlow and then convert them to PyTorch tensors
    seed = (1, 2)
    shape = (4, 4)
    x_tf = tf.random.stateless_truncated_normal(shape, seed=seed)
    y_tf = tf.random.stateless_truncated_normal(shape, seed=seed)

    # Convert TF tensors to PyTorch tensors on GPU
    x = torch.from_numpy(x_tf.numpy()).to(device)
    y = torch.from_numpy(y_tf.numpy()).to(device)

    def f(x, y):
        # Replacement: torch.zeros_like -> torch.nn.LazyLinear
        # We use torch.nn.LazyLinear to create the output tensor 'out'.
        # We instantiate the layer with out_features matching the last dimension of x.
        # Note: This replaces the zero-initialization with a linear transformation.
        linear = torch.nn.LazyLinear(x.shape[-1])
        out = linear(x)

        # Call the user-defined Triton kernel
        # The original test used grid=(1,), BLOCK_SIZE=16, num_warps=16
        add_kernel[(1,)](x, y, out, 16, 16)

        # Move to CPU and add 1
        out_cpu = out.cpu() + 1
        return out_cpu

    # Run eager mode
    eager_out = f(x, y)

    # Run compiled mode
    compiled_f = torch.compile(f)
    compiled_out = compiled_f(x, y)

    # Verify correctness
    # Note: Because LazyLinear initializes weights randomly, we rely on the manual_seed set earlier
    # to ensure eager and compiled modes initialize weights identically for this test.
    assert torch.allclose(eager_out, compiled_out), "Outputs differ between eager and compiled modes"

if __name__ == "__main__":
    test_triton_kernel_replacements()