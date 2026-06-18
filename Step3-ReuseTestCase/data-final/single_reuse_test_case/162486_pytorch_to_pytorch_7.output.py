import torch
import torch.nn as nn

def test_rnn_with_default_device(device: str = 'cuda'):
    # Set the default device, similar to the original bug report
    torch.set_default_device(device)

    # Define RNN parameters
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Instantiate RNN. It should pick up the default device.
    # This replaces the dataset creation in the original test.
    rnn = nn.RNN(input_size, hidden_size, num_layers)

    # Create input data. It should pick up the default device.
    # L = sequence length, N = batch size
    L, N = 5, 3
    input = torch.randn(L, N, input_size)
    h0 = torch.randn(num_layers, N, hidden_size)

    # Forward pass - this replaces the random_split call
    output, hn = rnn(input, h0)

    # Verify that the model and tensors are on the expected device
    assert rnn.weight_ih_l0.device.type == device, f"Model not on {device}"
    assert input.device.type == device, f"Input not on {device}"
    assert output.device.type == device, f"Output not on {device}"

    print(f"Device {device} worked for torch.nn.RNN.")

# Test with CPU (should always work)
test_rnn_with_default_device('cpu')

# Test with CUDA (if available)
if torch.cuda.is_available():
    test_rnn_with_default_device('cuda')
else:
    print("CUDA not available, skipping CUDA test.")