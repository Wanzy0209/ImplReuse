import torch
import torch.nn as nn

def f(input_tensor, h0):
    # Using nn.RNN which is a subclass of torch.nn.RNNBase
    # We instantiate it here to ensure the call is part of the compiled graph
    rnn = nn.RNN(input_size=1, hidden_size=1)
    output, hn = rnn(input_tensor, h0)
    return output

def backend(gm, inps):
    gm.print_readable()
    return gm

with torch.device("cuda"):
    # Inputs for RNN: (seq_len, batch, input_size)
    input_tensor = torch.randn(5, 1, 1, device="cuda")
    # Hidden state: (num_layers, batch, hidden_size)
    h0 = torch.randn(1, 1, 1, device="cuda")

    # Eager execution
    f(input_tensor, h0)

    # Compiled execution
    # This verifies if torch.compile under torch.device context
    # affects torch.nn.RNNBase similarly to torch.split
    torch.compile(f, backend=backend)(input_tensor, h0)