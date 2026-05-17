import torch
import numpy as np

torch.manual_seed(0)
torch._inductor.config.fallback_random = True

def foo(sequences):
    # Adapted to use torch.nn.utils.rnn.pad_sequence
    # Replaces torch.nn.functional.interpolate
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )
    # Adapted post-processing: original did squeeze -> argmin.
    # Here we sum over the sequence length to ensure the compiler handles
    # the padding logic correctly before reduction.
    result = padded.sum(dim=1)
    return result

np.random.seed(0)
# Generate input: a list of tensors with different lengths
# Using float64 to match the original test case's precision sensitivity
seq1 = torch.from_numpy(np.random.uniform(0, 10, size=(10, 5))).to(torch.float64)
seq2 = torch.from_numpy(np.random.uniform(0, 10, size=(15, 5))).to(torch.float64)
seq3 = torch.from_numpy(np.random.uniform(0, 10, size=(8, 5))).to(torch.float64)
input_list = [seq1, seq2, seq3]

cfoo = torch.compile(foo)
eager_res = foo(input_list)
compile_res = cfoo(input_list)

torch.testing.assert_close(eager_res, compile_res)