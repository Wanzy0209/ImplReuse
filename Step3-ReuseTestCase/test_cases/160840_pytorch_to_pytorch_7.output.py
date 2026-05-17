import torch
import torch.nn.utils.rnn as rnn_utils

torch.manual_seed(0)

# Prepare inputs for pack_padded_sequence
# Input shape: (Sequence Length, Batch Size, Feature Size)
# We create a tensor of shape (10, 3, 2) with valid lengths [10, 8, 5]
x = torch.randn(10, 3, 2)
lengths = torch.tensor([10, 8, 5])

def foo(input, lengths):
    # Call the similar API: torch.nn.utils.rnn.pack_padded_sequence
    packed = rnn_utils.pack_padded_sequence(
        input,
        lengths=lengths,
        batch_first=False,
        enforce_sorted=True
    )
    return packed

cfoo = torch.compile(foo)
eager_res = foo(x, lengths)
compile_res = cfoo(x, lengths)

# Verify consistency between eager and compiled results
# PackedSequence comparison requires checking data and batch_sizes
torch.testing.assert_close(eager_res.data, compile_res.data)
torch.testing.assert_close(eager_res.batch_sizes, compile_res.batch_sizes)