import torch
import torch.nn as nn

# Replicate the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Check for CUDA availability as the bug report uses CUDA
if not torch.cuda.is_available():
    print("CUDA not available, skipping test.")
    exit()

torch.manual_seed(70609)

# Adapted test case for torch.nn.RNNBase
# The original issue involved eager/compile divergence in matmul decomposition with float16.
# RNNBase (used by RNN, LSTM, GRU) relies heavily on internal matmul operations.
def test_rnn_base_divergence():
    # Define a simple RNN model (RNNBase subclass)
    # Dimensions chosen to reflect the matrix sizes involved in the bug report (e.g., 6, 13)
    # input_size=13, hidden_size=6
    class SimpleRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_size=13, hidden_size=6, num_layers=1, dtype=torch.float16)

        def forward(self, x):
            return self.rnn(x)

    # Instantiate model and move to CUDA
    model = SimpleRNN().cuda()
    
    # Compile the model
    compiled_model = torch.compile(model)

    # Create input data
    # Shape: (seq_len, batch, input_size) -> (14, 1, 13)
    # Using sequence length 14 to match dimensions in the original bug report
    input_tensor = torch.randn(14, 1, 13, dtype=torch.float16, device='cuda')

    # Run Eager
    with torch.no_grad():
        out_eager, h_eager = model(input_tensor)

    # Run Compiled
    with torch.no_grad():
        out_compiled, h_compiled = compiled_model(input_tensor)

    # Check for divergence
    # Using a tolerance suitable for float16 operations
    try:
        torch.testing.assert_close(out_eager, out_compiled, rtol=1e-2, atol=1e-2)
        print("Test passed: No divergence detected between Eager and Compiled RNN.")
    except AssertionError as e:
        print(f"Test failed: Divergence detected.\n{e}")
        raise

if __name__ == "__main__":
    test_rnn_base_divergence()